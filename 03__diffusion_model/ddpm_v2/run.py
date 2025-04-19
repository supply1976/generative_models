import os, sys, argparse, shutil, logging
import math, time, datetime, yaml
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
# 
import modelDef
from data_loader import DataLoader

tf.debugging.disable_traceback_filtering()


def init_logging(filename, checkpoint=None):
  # mode = "a+" if checkpoint is not None else "w+"
  mode = "w+"
  logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(message)s",
    format="%(message)s",
    handlers=[logging.FileHandler(filename, mode=mode), logging.StreamHandler()],
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--training", dest='training', action='store_true')
  parser.add_argument("--imgen", dest='imgen', action='store_true')
  parser.add_argument("--clip_denoise", action='store_true')
  FLAGS, _ = parser.parse_known_args()
  
  with open(FLAGS.config, 'r') as f:
    config_dict = yaml.safe_load(f)

  dataset_dict   = config_dict['DATASET']
  training_dict  = config_dict['TRAINING']
  imgen_dict     = config_dict['IMAGE_GENERATION']

  # dataset
  dataset_name   = dataset_dict['NAME']
  dataset_path   = dataset_dict['PATH']
  if 'REPEAT' not in dataset_dict.keys(): 
    dataset_dict['REPEAT'] = 1
  dataset_repeat = dataset_dict['REPEAT']
  clip_size = dataset_dict['PREPROCESSING']['CLIP_SIZE']
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
  num_resnet_blocks  = training_dict['NETWORK']['NUM_RESNET_BLOCKS']
  block_size         = training_dict['NETWORK']['BLOCK_SIZE']
  norm_groups        = training_dict['NETWORK']['NORM_GROUPS']
  first_channel      = training_dict['NETWORK']['FIRST_CHANNEL']
  channel_multiplier = training_dict['NETWORK']['CHANNEL_MULTIPLIER']
  has_attention      = training_dict['NETWORK']['HAS_ATTENTION']
  assert len(channel_multiplier)==len(has_attention)
  widths = [first_channel * mult for mult in channel_multiplier]
  # hyper-parameters
  epochs        = training_dict['HYPER_PARAMETERS']['EPOCHS']
  batch_size    = training_dict['HYPER_PARAMETERS']['BATCH_SIZE']
  learning_rate = training_dict['HYPER_PARAMETERS']['LEARNING_RATE']
  # for image generation (imgen)
  imgen_model_path = imgen_dict['MODEL_PATH']
  num_gen_images   = imgen_dict['NUM_GEN_IMAGES']
  export_interm    = imgen_dict['EXPORT_INTERM']
  reverse_stride   = imgen_dict['REVERSE_STRIDE']
  gen_inputs       = imgen_dict['GEN_INPUTS']
  gen_output_dir   = imgen_dict['GEN_OUTPUT_DIR']
  ddim_eta         = imgen_dict['DDIM_ETA']
  random_seed      = imgen_dict['RANDOM_SEED']

  # GPU devices
  gpus = tf.config.list_physical_devices("GPU")
  if gpus: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

  # Build (initialize) the unet model
  network = modelDef.build_model(
    image_size    = input_image_size, 
    image_channel = input_image_channel,
    widths = widths,
    has_attention = has_attention,
    num_resnet_blocks = num_resnet_blocks,
    norm_groups = norm_groups,
    activation_fn = keras.activations.swish,
    block_size = block_size,
    )
  
  ema_network = modelDef.build_model(
    image_size    = input_image_size, 
    image_channel = input_image_channel,
    widths = widths,
    has_attention = has_attention,
    num_resnet_blocks = num_resnet_blocks,
    norm_groups = norm_groups,
    activation_fn = keras.activations.swish,
    block_size = block_size,
    )

  network.summary()
  # Initially the weights are the same
  ema_network.set_weights(network.get_weights())  

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
    model_nametag = model_nametag+"rb"+str(num_resnet_blocks)
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
        try:
          ddpm.load_weights(trained_h5)
        except:
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
    if os.path.isdir(dataset_path):
      logging.info("[INFO] Use a folder contains multiple npz files for training")
      logging.info("[INFO] Dataset path: {}".format(dataset_path))
      dataloader = DataLoader(
        data_dir=dataset_path, 
        crop_size=clip_size,
        dataset_repeat=dataset_repeat,
        )
      train_ds, valid_ds = dataloader.load_dataset()
    else:
      assert os.path.isfile(dataset_path)
      assert dataset_path.endswith(".npz")
      logging.info("[INFO] Use single (big) npz file for training")
      all_images = np.load(dataset_path)['images']
      all_images = 2*(all_images) - 1.0  ; # (0, 1) -> (-1, 1)
      idx = np.arange(len(all_images))
      np.random.shuffle(idx)
      num_val = int(0.1*len(all_images))
      train_ds = tf.data.Dataset.from_tensor_slices(all_images)
      train_ds = train_ds.cache().repeat(dataset_repeat)
      train_ds = train_ds.shuffle(train_ds.cardinality())
      valid_ds = tf.data.Dataset.from_tensor_slices(all_images[0:num_val])
      valid_ds = valid_ds.cache()
    
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
      
    # get input image shape (preprocessed)
    for x in train_ds.take(1):
      _, h, w, c = x.shape
      logging.info("dataset one batch info: {}".format(x.shape))
      logging.info("signal rescale to: ({},{})".format(x.numpy().min(), x.numpy().max()))
    assert c==input_image_channel
    assert h==input_image_size
    
    callback_save_ema_latest = keras.callbacks.LambdaCallback(
      on_epoch_end=lambda epoch,logs: ddpm.save_model(epoch, savedir=logging_dir))
    callback_genimages = keras.callbacks.LambdaCallback(
      on_train_end=ddpm.generate_images)
    
    logging.info("[INFO] Forward Training Steps: {}".format(timesteps))
    logging.info("[INFO] Scheduler: {} ".format(scheduler))
    logging.info("[INFO] Learning Rate: {}".format(learning_rate))
    logging.info("[INFO] Predict Type: {}".format(pred_type))
    logging.info("[INFO] Loss Function: {}".format(loss_fn))
    logging.info("[INFO] Total Epochs: {}".format(epochs))
    logging.info("[INFO] Dataset Repeat: {}".format(dataset_repeat))

    csv_logger = CSVLogger(os.path.join(logging_dir, "log.csv"), append=True, separator=",")
    best_ckpt_path = os.path.join(logging_dir, "dm_best.weights.h5")
    callback_save_ema_best = keras.callbacks.ModelCheckpoint(
      filepath=best_ckpt_path,
      save_weights_only=True,
      monitor='val_loss',
      mode='min',
      save_best_only=True,
      )
    callback_list = [
      csv_logger, 
      callback_save_ema_latest,
      callback_save_ema_best,
      callback_genimages,
      ]

    if loss_fn == "MAE":
      loss_fn = keras.losses.MeanAbsoluteError()
    elif loss_fn == 'MSE':
      loss_fn = keras.losses.MeanSquaredError()
    else:
      raise NotImplementedError
    # Compile the model
    ddpm.compile(
      loss=loss_fn,
      optimizer=keras.optimizers.Adam(
        learning_rate=learning_rate,
        #weight_decay=1.0e-5,
        )
    )

    # Train the model
    ddpm.fit(
      train_ds,
      validation_data=valid_ds,
      epochs=epochs,
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
    try:
      ddpm.load_weights(imgen_model_path)
    except:
      ddpm.ema_network.load_weights(imgen_model_path)
      ddpm.network.set_weights(ddpm.ema_network.get_weights())

    if gen_output_dir is None: 
      gen_steps = str(diff_util_infer.timesteps // diff_util_infer.reverse_stride)+"steps"
      gen_dir = "_".join(["imgen",gen_steps,gen_date])
      gen_dir = os.path.join(model_dir, gen_dir)
      os.mkdir(gen_dir)
    else:
      gen_dir = gen_output_dir
      if not os.path.isdir(gen_dir): os.mkdir(gen_dir)
    
    init_logging(os.path.join(gen_dir, "imgen.log"))
    logging.info("[IMGEN] Start to generate images using model: {}".format(imgen_model_path))
    logging.info("[IMGEN] model predict type: {}".format(diff_util_infer.pred_type))
    logging.info("[IMGEN] DDIM eta = {}".format(diff_util_infer.ddim_eta))
    logging.info("[IMGEN] set random seed: {}".format(random_seed))
    tf.random.set_seed(random_seed)

    if gen_inputs is None:
      logging.info("Generating Images from pure random noise")
      pass

    elif os.path.isfile(gen_inputs) and gen_inputs.endswith('npz'):
      logging.info("Use external .npz file as start to generate images")
      logging.info("Generating Images from {}".format(gen_inputs))
      data = np.load(gen_inputs)['images']
      gen_inputs = tf.convert_to_tensor(data, tf.float32)
    
    elif gen_inputs == "identical_noise":
      logging.info("Generating Images from identical noise")
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
    
    elif gen_inputs == "custom2":
      pass

    else:
      return

    clip_denoise = True if FLAGS.clip_denoise else False

    t0 = time.time()
    ddpm.generate_images(
      num_images=num_gen_images, 
      savedir=gen_dir,
      save_ini=False,
      _freeze=False,
      gen_inputs=gen_inputs,
      clip_denoise=clip_denoise,
      export_interm=export_interm)
    deltaT = np.around((time.time()-t0), 1)
    logging.info("Generated {} images with {} seconds".format(num_gen_images, deltaT))

     
  else:
    ddpm.summary()
    print("no action")


if __name__=="__main__":
  main()
