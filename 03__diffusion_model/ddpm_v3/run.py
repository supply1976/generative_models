import os
import argparse
import shutil
import logging
import datetime
import time
import yaml
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tqdm import tqdm

import modelDef
from data_loader import DataLoader


#tf.config.optimizer.set_jit(True) # enable XLA
#tf.debugging.disable_traceback_filtering()

# enable_op_determinism() will turn off XLA, and slow down the training time
#tf.config.experimental.enable_op_determinism()
#tf.random.set_seed(1)


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, base_lr, total_steps, warmup_steps, min_lr=0.0):
    super().__init__()
    self.base_lr = base_lr
    self.total_steps = total_steps
    self.warmup_steps = warmup_steps
    self.min_lr = min_lr

  def __call__(self, step):
    # Linear warmup
    warmup_lr = self.base_lr * tf.cast(step, tf.float32)/tf.cast(self.warmup_steps, tf.float32)
    # Cosine decay
    cosine_steps = tf.cast(step - self.warmup_steps, tf.float32)
    cosine_total = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
    cosine_decay = 0.5 * (1 + tf.cos(np.pi * cosine_steps / cosine_total))
    decayed = (self.base_lr - self.min_lr) * cosine_decay + self.min_lr
    lr = tf.where(step < self.warmup_steps, warmup_lr, decayed)
    return lr


class TQDMProgressBar(keras.callbacks.Callback):
  def on_train_begin(self, logs=None):
    self.epochs = self.params['epochs']
    self.steps_per_epoch = self.params['steps']

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch = epoch
    self.pbar = tqdm(total=self.steps_per_epoch,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=False)

  def on_train_batch_end(self, batch, logs=None):
    # global step
    gs = int(self.model.optimizer.iterations.numpy())
    lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
    self.pbar.set_postfix({
        'lr': f"{lr:.4e}",
        'gs': gs,
        })
    self.pbar.update(1)

  def on_epoch_end(self, epoch, logs=None):
    self.pbar.close()


def init_logging(filename, checkpoint=None):
  # mode = "a+" if checkpoint is not None else "w+"
  mode = "w+"
  logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(message)s",
    format="%(message)s",
    handlers=[logging.FileHandler(filename, mode=mode), logging.StreamHandler()],
    )


def parse_config(config_path):
  with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
  return cfg['DATASET'], cfg['TRAINING'], cfg['IMAGE_GENERATION']


def build_models(image_size, image_channel, widths, has_attention,
                 num_heads, num_res_blocks, norm_groups, block_size, temb_dim,
                 dropout_rate=0.0, kernel_size=3, use_cross_attention=False):
  kwargs = dict(
      image_size=image_size,
      image_channel=image_channel,
      widths=widths,
      has_attention=has_attention,
      num_heads=num_heads,
      num_res_blocks=num_res_blocks,
      norm_groups=norm_groups,
      actf=keras.activations.swish,
      block_size=block_size,
      temb_dim=temb_dim,
      dropout_rate=dropout_rate,
      kernel_size=kernel_size,
      use_cross_attention=use_cross_attention,
  )
  network = modelDef.build_model(**kwargs)
  ema_network = modelDef.build_model(**kwargs)
  ema_network.set_weights(network.get_weights())
  return network, ema_network


def prepare_datasets(dataset_path, img_size, batch_size, crop_size=None, dtype=tf.float32):
  if os.path.isdir(dataset_path):
    dataloader = DataLoader(
        data_dir=dataset_path,
        img_size=img_size,
        crop_size=crop_size,
    )
    train_ds, valid_ds = dataloader._get_dataset()
  else:
    all_images = np.load(dataset_path)['images']
    all_images = 2 * all_images - 1.0
    idx = np.arange(len(all_images))
    np.random.shuffle(idx)
    num_val = int(0.1 * len(all_images))
    train_ds = tf.data.Dataset.from_tensor_slices(all_images)
    train_ds = train_ds.cache().shuffle(buffer_size=10000).repeat()
    valid_ds = tf.data.Dataset.from_tensor_slices(all_images[0:num_val]).cache()

  train_ds = train_ds.batch(batch_size, drop_remainder=True)
  train_ds = train_ds.map(lambda x: tf.cast(x, dtype), num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
  valid_ds = valid_ds.batch(batch_size)
  valid_ds = valid_ds.map(lambda x: tf.cast(x, dtype), num_parallel_calls=tf.data.AUTOTUNE)
  valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
  return train_ds, valid_ds


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--training", dest='training', action='store_true')
  parser.add_argument("--imgen", dest='imgen', action='store_true')
  parser.add_argument("--clip_denoise", action='store_true')
  parser.add_argument("--mixed_precision", action='store_true',
                      help='Enable mixed float16 training')
  parser.add_argument("--enable_xla", action='store_true',
                      help='Enable XLA JIT compilation')
  FLAGS, _ = parser.parse_known_args()

  if FLAGS.mixed_precision:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
  if FLAGS.enable_xla:
    tf.config.optimizer.set_jit(True)

  dtype = tf.float16 if FLAGS.mixed_precision else tf.float32

  dataset_dict, training_dict, imgen_dict = parse_config(FLAGS.config)

  # dataset
  dataset_name   = dataset_dict['NAME']
  dataset_path   = dataset_dict['PATH']
  crop_size = dataset_dict['PREPROCESSING']['CROP_SIZE']
  CLIP_MIN  = dataset_dict['PREPROCESSING']['CLIP_MIN']
  CLIP_MAX  = dataset_dict['PREPROCESSING']['CLIP_MAX']

  # input shape
  input_image_size    = training_dict['INPUT_IMAGE_SIZE']
  input_image_channel = training_dict['INPUT_IMAGE_CHANNEL']
  # training
  training_output_dir = training_dict['OUTPUT_DIR']
  if not os.path.isdir(training_output_dir): os.mkdir(training_output_dir)
  is_new_train       = training_dict['IS_NEW_TRAIN']
  trained_h5         = training_dict['TRAINED_H5']
  pred_type          = training_dict['PRED_TYPE']
  loss_fn            = training_dict['LOSS_FN']
  # network
  scheduler          = training_dict['NETWORK']['SCHEDULER']
  timesteps          = training_dict['NETWORK']['TIMESTEPS']
  num_res_blocks     = training_dict['NETWORK']['NUM_RES_BLOCKS']
  block_size         = training_dict['NETWORK']['BLOCK_SIZE']
  norm_groups        = training_dict['NETWORK']['NORM_GROUPS']
  first_channel      = training_dict['NETWORK']['FIRST_CHANNEL']
  channel_multiplier = training_dict['NETWORK']['CHANNEL_MULTIPLIER']
  has_attention      = training_dict['NETWORK']['HAS_ATTENTION']
  num_heads          = training_dict['NETWORK']['NUM_HEADS']
  assert len(channel_multiplier)==len(has_attention)
  widths = [first_channel * mult for mult in channel_multiplier]
  temb_dim           = training_dict['NETWORK']['TIME_EMB_DIM']
  dropout_rate       = training_dict['NETWORK'].get('DROPOUT_RATE', 0.0)
  kernel_size        = training_dict['NETWORK'].get('KERNEL_SIZE', 3)
  use_cross_attention = training_dict['NETWORK'].get('USE_CROSS_ATTENTION', False)

  # hyper-parameters
  epochs          = training_dict['HYPER_PARAMETERS']['EPOCHS']
  batch_size      = training_dict['HYPER_PARAMETERS']['BATCH_SIZE']
  lr_type         = training_dict['HYPER_PARAMETERS']['LR_TYPE']
  learning_rate   = training_dict['HYPER_PARAMETERS']['LEARNING_RATE']
  warmup_steps    = training_dict['HYPER_PARAMETERS']['WARMUP_STEPS']
  steps_per_epoch = training_dict['HYPER_PARAMETERS']['STEPS_PER_EPOCH']
  total_steps = None if steps_per_epoch is None else epochs * steps_per_epoch
  # for image generation (imgen)
  imgen_model_path = imgen_dict['MODEL_PATH']
  num_gen_images   = imgen_dict['NUM_GEN_IMAGES']
  export_interm    = imgen_dict['EXPORT_INTERM']
  reverse_stride   = imgen_dict['REVERSE_STRIDE']
  gen_inputs       = imgen_dict['GEN_INPUTS']
  gen_output_dir   = imgen_dict['GEN_OUTPUT_DIR']
  ddim_eta         = imgen_dict['DDIM_ETA']
  random_seed      = imgen_dict['RANDOM_SEED']
  if "_FREEZE_INI" not in imgen_dict.keys():
    imgen_dict['_FREEZE_INI'] = False
  _freeze_ini      = imgen_dict['_FREEZE_INI']

  # GPU devices
  gpus = tf.config.list_physical_devices("GPU")
  if gpus: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

  # Build (initialize) the unet model
  network, ema_network = build_models(
    image_size=input_image_size,
    image_channel=input_image_channel,
    widths=widths,
    has_attention=has_attention,
    num_heads=num_heads,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    block_size=block_size,
    temb_dim=temb_dim,
    dropout_rate=dropout_rate,
    kernel_size=kernel_size,
    use_cross_attention=use_cross_attention,
  )

  network.summary()

  # Get an instance of the Gaussian Diffusion utilities
  # util for training
  diff_util_train = modelDef.DiffusionUtility(
    b0=0.1, b1=20, timesteps=timesteps, 
    scheduler=scheduler, pred_type=pred_type, reverse_stride=1
    )
  # util for inference
  diff_util_infer = modelDef.DiffusionUtility(
    b0=0.1, b1=20, timesteps=timesteps, 
    scheduler=scheduler, pred_type=pred_type, reverse_stride=reverse_stride,
    ddim_eta=ddim_eta,
    )
   
  if FLAGS.training: 
    # Get the diffusion model (keras.Model)
    ddpm = modelDef.DiffusionModel(
      network=network, 
      ema_network=ema_network,
      diff_util=diff_util_train,
      timesteps=timesteps,
      )
    
    assert dataset_name is not None
    
    (train_ds, valid_ds) = (None, None)
    input_shape = (input_image_size, input_image_size, input_image_channel)
    # create folder for dataset tag
    dataset_tag = os.path.join(
      os.path.abspath(training_output_dir), 
      "{}_{}x{}x{}".format(dataset_name, *input_shape)
      )
    if not os.path.isdir(dataset_tag): os.mkdir(dataset_tag)
    # create model nametag
    model_nametag = "unet"+str(first_channel)+"m"+"".join(map(str, channel_multiplier))
    model_nametag = model_nametag+"g"+str(norm_groups)
    model_nametag = model_nametag+"rb"+str(num_res_blocks)
    model_nametag = model_nametag+"bk"+str(block_size)
    tr_output_dir = os.path.join(dataset_tag, model_nametag)
    if not os.path.isdir(tr_output_dir): os.mkdir(tr_output_dir)

    dateID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dateID = "_".join([scheduler,str(timesteps),pred_type, loss_fn, dateID])
    
    if is_new_train:
      logging_dir = os.path.join(tr_output_dir, dateID)
      os.mkdir(logging_dir)
      init_logging(os.path.join(logging_dir, "train.log"))
      logging.info("[INFO] Start a new training")
    else: 
      if trained_h5 is None:
        logging.info('if "IS_NEW_TRAIN: false", you must provide a trained .h5 file')
        return 
      else:
        # load .h5 file
        trained_h5 = os.path.abspath(trained_h5)
        restored_model_ID = os.path.basename(os.path.dirname(trained_h5))
        restored_model_ID = restored_model_ID.split("_")[-1]
        ddpm.ema_network.load_weights(trained_h5)
        ddpm.network.set_weights(ddpm.ema_network.get_weights())

      logging_dir = os.path.join(tr_output_dir, dateID+"_from_"+restored_model_ID)
      os.mkdir(logging_dir)
      init_logging(os.path.join(logging_dir,"train.log"))
      logging.info("[INFO] Restoring model from: {}".format(trained_h5))
      logging.info("[INFO] Continuous Transfer training ...")
    
    shutil.copy(FLAGS.config, os.path.join(logging_dir, "training_config.yaml"))
    # end of creating logging_dir 

    logging.info("[INFO] Training Start Time: {}".format(datetime.datetime.now()))
    t0 = time.time()
    
    assert dataset_path is not None
    logging.info("[INFO] User defined dataset name:{}".format(dataset_name))
    train_ds, valid_ds = prepare_datasets(
      dataset_path,
      img_size=input_image_size,
      batch_size=batch_size,
      crop_size=crop_size,
      dtype=dtype,
    )
      
    # get input image shape
    for x in train_ds.take(1):
      _, h, w, c = x.shape
      logging.info("dataset one batch info: {}".format(x.shape))
      logging.info("signal rescale to: ({},{})".format(x.numpy().min(), x.numpy().max()))
    assert c==input_image_channel
    assert h==input_image_size
    
    callback_save_ema_latest = keras.callbacks.LambdaCallback(
      on_epoch_end=lambda epoch, logs: ddpm.save_model(epoch, savedir=logging_dir))
    callback_genimages = keras.callbacks.LambdaCallback(
      on_train_end=ddpm.generate_images)
    
    logging.info("[INFO] Forward Training Steps: {}".format(timesteps))
    logging.info("[INFO] Scheduler: {} ".format(scheduler))
    logging.info("[INFO] Learning Rate: {}".format(learning_rate))
    logging.info("[INFO] Predict Type: {}".format(pred_type))
    logging.info("[INFO] Loss Function: {}".format(loss_fn))
    logging.info("[INFO] Total Epochs: {}".format(epochs))

    csv_logger = CSVLogger(os.path.join(logging_dir, "log.csv"), append=True, separator=",")
    #best_ckpt_path = os.path.join(logging_dir, "dm_best.weights.h5")
    #callback_save_ema_best = keras.callbacks.ModelCheckpoint(
    #  filepath=best_ckpt_path,
    #  save_weights_only=True,
    #  monitor='val_loss',
    #  mode='min',
    #  save_best_only=True,
    #  )
    
    callback_list = [
      csv_logger, 
      callback_save_ema_latest,
      TQDMProgressBar(),
      #callback_save_ema_best,
      callback_genimages,
      ]

    if loss_fn == "MAE":
      loss_fn = keras.losses.MeanAbsoluteError()
    elif loss_fn == 'MSE':
      loss_fn = keras.losses.MeanSquaredError()
    else:
      raise NotImplementedError
    

    if lr_type == 'constant':
      lr_schedule = learning_rate

    elif lr_type == 'warmup_cosine':
      assert warmup_steps is not None
      lr_schedule = WarmUpCosine(
        base_lr = learning_rate,
        warmup_steps = warmup_steps,
        total_steps = total_steps,
        )

    elif lr_type == 'cosine_decay':
      lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = learning_rate,
        decay_steps = 10000,
        alpha = 0.0,
        )
    else:
      raise NotImplementedError

    optimizer = keras.optimizers.AdamW(
      learning_rate = lr_schedule,
      #weight_decay=1.0e-5,
      )

    # Compile the model
    ddpm.compile(loss=loss_fn,optimizer=optimizer)

    # Train the model
    ddpm.fit(
      train_ds,
      validation_data=valid_ds,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      callbacks=callback_list,
      )
    deltaT = np.around((time.time() - t0)/3600.0, 4)
    nowT = datetime.datetime.now()
    logging.info("[INFO] Training End: {}, elapsed time: {} hours".format(nowT, deltaT))

  elif FLAGS.imgen:
    assert imgen_model_path is not None
    assert os.path.isfile(imgen_model_path)
    model_dir = os.path.dirname(imgen_model_path)
    gen_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    ddpm = modelDef.DiffusionModel(
      network=network, 
      ema_network=ema_network,
      diff_util=diff_util_infer,
      timesteps=timesteps,
      )

    ddpm.ema_network.load_weights(imgen_model_path)

    if gen_output_dir is None: 
      gen_steps = str(diff_util_infer.timesteps // diff_util_infer.reverse_stride)+"steps"
      gen_dir ="_".join([
        "imgen", gen_steps, "tf"+tf.__version__, os.uname().nodename, gen_date])
      gen_dir = os.path.join(model_dir, gen_dir)
      os.mkdir(gen_dir)
    else:
      gen_dir = gen_output_dir
      if not os.path.isdir(gen_dir): os.mkdir(gen_dir)
    
    init_logging(os.path.join(gen_dir, "imgen.log"))
    logging.info("[IMGEN] Start to generate images using model: {}".format(imgen_model_path))
    logging.info("[IMGEN] model predict type: {}".format(diff_util_infer.pred_type))
    logging.info("[IMGEN] DDIM eta = {}".format(diff_util_infer.ddim_eta))
    logging.info("[IMGEN] Set random seed: {}".format(random_seed))
    logging.info("[IMGEN] clip_denoise: {}".format(FLAGS.clip_denoise))
    logging.info("[IMGEN] hostname: {}".format(os.uname().nodename))
    logging.info("[IMGEN] TF version: {}".format(tf.__version__))

    tf.random.set_seed(random_seed)

    if gen_inputs is None:
      logging.info("Generating Images from standard Gaussian noise")
      pass

    elif os.path.isfile(gen_inputs) and gen_inputs.endswith('npz'):
      logging.info("Use external .npz file as start to generate images")
      logging.info("Generating Images from {}".format(gen_inputs))
      data = np.load(gen_inputs)['images']
      gen_inputs = tf.identity(data, tf.float32)
    
    elif gen_inputs == "identical_noise":
      logging.info("Generating Images from identical Gaussian noise")
      _shape=(1, input_image_size, input_image_size, input_image_channel)
      gen_inputs = tf.random.normal(shape=_shape, dtype=tf.float32)
      gen_inputs = tf.tile(gen_inputs, [num_gen_images, 1,1,1])

    elif gen_inputs == "custom":
      _shape=(1, input_image_size, input_image_size, input_image_channel)
      tf.random.set_seed(1)
      z1 = tf.random.normal(shape=_shape, dtype=tf.float32)
      tf.random.set_seed(2)
      z2 = tf.random.normal(shape=_shape, dtype=tf.float32)
      theta = np.linspace(0, 1, 11)
      v = np.cos(theta*np.pi/2)
      z = (v[:,None,None,None])*z1 + np.sqrt((1-v[:,None,None,None]**2))*z2
      gen_inputs = z

    elif gen_inputs == "gen_random_seed_map":
      _shape=(1, input_image_size, input_image_size, input_image_channel)
      zlist = []
      for seed in range(10):
        tf.random.set_seed(seed)
        zlist.append(tf.random.normal(shape=_shape, dtype=tf.float32))
      gen_inputs = tf.concat(zlist, axis=0)
    
    elif gen_inputs=="_freeze_ini":
      images = np.load("/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/metal_test1/pitch_8_512x512x1/all_images_713x512x512x1.npz")['images']
      images = 2*images -1.0
      images = images[0:10]
      n, h, w, c = images.shape
      noises = tf.random.normal(shape=images.shape, dtype=tf.float32)
      noises = noises.numpy()
      noises[:, 0:h//2, 0:h//2, :] = images[:, 0:h//2, 0:w//2, :]
      gen_inputs = noises

    else:
      return

    clip_denoise = True if FLAGS.clip_denoise else False

    t0 = time.time()

    ddpm.generate_images(
      num_images=num_gen_images, 
      savedir=gen_dir,
      gen_inputs=gen_inputs,
      _freeze_ini=_freeze_ini,
      clip_denoise=clip_denoise,
      export_interm=export_interm)
    #
    deltaT = np.around((time.time()-t0), 1)
    logging.info("Generated {} images with {} seconds".format(num_gen_images, deltaT))

  else:
    print(network.get_config())
    print("no action")


if __name__=="__main__":
  main()
