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
  FLAGS, _ = parser.parse_known_args()
  
  with open(FLAGS.config, 'r') as f:
    config_dict = yaml.safe_load(f)

  dataset_dict   = config_dict['DATASET']
  training_dict  = config_dict['TRAINING']
  imgen_dict     = config_dict['IMAGE_GENERATION']

  # dataset
  dataset_name = dataset_dict['NAME']
  dataset_path = dataset_dict['PATH']
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
  # network
  scheduler          = training_dict['NETWORK']['SCHEDULER']
  timesteps          = training_dict['NETWORK']['TIMESTEPS']
  num_resnet_blocks  = training_dict['NETWORK']['NUM_RESNET_BLOCKS']
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
  # imgen
  imgen_model_path = imgen_dict['MODEL_PATH']
  num_gen_images   = imgen_dict['NUM_GEN_IMAGES']
  given_samples    = imgen_dict['GIVEN_SAMPLES']
  export_interm    = imgen_dict['EXPORT_INTERM']
  prev_n_step      = imgen_dict['PREV_N_STEP']

  # GPU devices
  gpus = tf.config.list_physical_devices("GPU")
  if gpus: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

  # Build (initialize) the unet model
  network = modelDef.build_model(
    image_size    = input_image_size, 
    image_channel = input_image_channel,
    first_channel = first_channel,
    widths = widths,
    has_attention = has_attention,
    num_resnet_blocks = num_resnet_blocks,
    norm_groups = norm_groups,
    activation_fn = keras.activations.swish)
  
  ema_network = modelDef.build_model(
    image_size    = input_image_size, 
    image_channel = input_image_channel,
    first_channel = first_channel,
    widths = widths,
    has_attention = has_attention,
    num_resnet_blocks = num_resnet_blocks,
    norm_groups = norm_groups,
    activation_fn = keras.activations.swish)

  network.summary()
  # Initially the weights are the same
  ema_network.set_weights(network.get_weights())  

  # Get an instance of the Gaussian Diffusion utilities
  diff_util = modelDef.DiffusionUtility(
    b0=0.1, b1=20, timesteps=timesteps, prev_n_step=prev_n_step, scheduler=scheduler)
   
  if FLAGS.training: 
    if dataset_name is None:
      logging.info("no training dataset name is provided, exit")
      return 
    train_ds = None
    dateID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    input_shape = (input_image_size, input_image_size, input_image_channel)
    nametag = "DDPM{}_{}x{}x{}".format(first_channel, *input_shape)
    nametag = nametag + "__trset_{}".format(dataset_name)
    tr_output_dir = os.path.join(os.path.abspath(training_output_dir), nametag)
    if not os.path.isdir(tr_output_dir): os.mkdir(tr_output_dir)
    dateID = "_".join([scheduler,str(timesteps),dateID])
    if is_new_train:
      logging_dir = os.path.join(tr_output_dir, dateID)
      os.mkdir(logging_dir)
      init_logging(os.path.join(logging_dir, "train.log"))
      logging.info("Start a new training")
      shutil.copy(FLAGS.config, os.path.join(logging_dir, "training_config.yaml"))
    else: 
      if trained_h5 is None:
        logging.info('if "IS_NEW_TRAIN: false", you must provide a trained .h5 file')
        return 
      else:
        # .h5 file
        trained_h5 = os.path.abspath(trained_h5)
        restored_model_dir = os.path.dirname(trained_h5)
        load_status = ema_network.load_weights(trained_h5)
        network.set_weights(ema_network.get_weights())
        #load_status.assert_consumed()
      logging_dir = os.path.join(restored_model_dir, "cont_tr_"+dateID)
      os.mkdir(logging_dir)
      init_logging(os.path.join(logging_dir, "train.log"))
      logging.info("restoring model from: {}".format(trained_h5))
      logging.info("continous training ...")
    # Get the diffusion model (keras.Model)
    ddpm = modelDef.DiffusionModel(
      network=network, 
      ema_network=ema_network,
      diff_util=diff_util,
      timesteps=timesteps)
    
    logging.info("Training Start Time: {}".format(datetime.datetime.now()))
    t0 = time.time()
    if dataset_name in ["cifar10", "mnist"]:
      # use internal keras dataset for quick testing
      # ignore dataset_path when use these two dataset_name
      logging.info("use internal keras dataset {} as training data".format(dataset_name))
      if dataset_name == "cifar10":
        (train_images, train_labels), _ = keras.datasets.cifar10.load_data()
        #ID, _ = np.where(train_labels==1)
        #train_images = train_images[ID]

      elif dataset_name == "mnist": 
        (train_images, _), _ = keras.datasets.mnist.load_data()
        train_images = np.expand_dims(train_images, axis=-1)

      else:
        train_images = None
      train_images = train_images.astype(np.float32)
      train_images = 2*(train_images / 255.0) - 1 ; # rescale to (-1, 1)
      if dataset_name=="mnist":
        train_images = tf.image.resize(train_images, [32, 32])
      train_ds = tf.data.Dataset.from_tensor_slices(train_images)
      train_ds = train_ds.cache()
      train_ds = train_ds.shuffle(train_ds.cardinality())

    else:
      # for other dataset_name, check the dataset_path
      if dataset_path is None:
        logging.info("Please provide dataset path, Exit")
        return
      else:
        logging.info("user defined dataset name:{}".format(dataset_name))
        if os.path.isdir(dataset_path):
          logging.info("use a folder contains multiple npz files for training")
          dataloader = DataLoader(data_dir=dataset_path, crop_size=clip_size)
          train_ds = dataloader.load_dataset()
        elif os.path.isfile(dataset_path):
          logging.info("use single (big) npz file for trainingg")
          data = np.load(dataset_path)
          train_images = data['images']
          train_ds = tf.data.Dataset.from_tensor_slices(train_images)
          train_ds = train_ds.cache()
          train_ds = train_ds.shuffle(train_ds.cardinality())
        else:
          return
    # make sure train dataset is prepared
    assert train_ds is not None
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    # get input image shape (preprocessed)
    for x in train_ds.take(1):
      _, h, w, c = x.shape
      logging.info("dataset one batch info: {}".format(x.shape))
      logging.info("signal rescale to: ({},{})".format(x.numpy().min(), x.numpy().max()))
    assert c==input_image_channel
    assert h==input_image_size
    
    callback_save_model = keras.callbacks.LambdaCallback(
      on_epoch_end=lambda epoch,logs: ddpm.save_model(epoch, savedir=logging_dir))
    callback_genimages = keras.callbacks.LambdaCallback(
      on_train_end=ddpm.generate_images)
    
    logging.info("forward time steps: {}".format(timesteps))
    logging.info("learning rate: {}".format(learning_rate))
    logging.info("epochs: {}".format(epochs))

    csv_logger = CSVLogger(os.path.join(logging_dir, "log.csv"), append=True, separator=",")
    callback_list = [csv_logger, callback_save_model, callback_genimages]

    # Compile the model
    ddpm.compile(
      loss=keras.losses.MeanSquaredError(),
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    # Train the model
    ddpm.fit(
      train_ds,
      epochs=epochs,
      callbacks=callback_list
      )
    #ema_network.save_weights(os.path.join(logging_dir, "ema_final.weights.h5"))
    deltaT = np.around((time.time() - t0)/3600.0, 4)
    nowT = datetime.datetime.now()
    logging.info("Training End: {}, elapsed time: {} hours".format(nowT, deltaT))
    
  elif FLAGS.imgen:
    assert imgen_model_path is not None
    assert os.path.isfile(imgen_model_path)
    model_dir = os.path.dirname(imgen_model_path)
    gen_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gen_dir = os.path.join(model_dir, "imgen_"+gen_date)
    os.mkdir(gen_dir)
    load_status = ema_network.load_weights(imgen_model_path)
    network.set_weights(ema_network.get_weights())
    ddpm = modelDef.DiffusionModel(
      network=network, 
      ema_network=ema_network,
      diff_util=diff_util,
      timesteps=timesteps)
    init_logging(os.path.join(gen_dir, "gen_images.log"))
    logging.info("start to generate images using model: {}".format(imgen_model_path))
    
    if given_samples is None:
      logging.info("use gaussian random noise to generate images")
      t0 = time.time()
      ddpm.generate_images(
        num_images=num_gen_images, savedir=gen_dir, export_interm=export_interm)
      deltaT = np.around((time.time()-t0)/3600, 4)
      logging.info("generate {} images with {} hours".format(num_gen_images, deltaT))
    else:
      # use external given images (.npz) as input to generate images
      data = np.load(given_samples)
      data = tf.convert_to_tensor(data['images'], tf.float32)
      ddpm.generate_images(
        given_samples=data, savedir=gen_dir, freeze_1st=True, export_interm=export_interm)

  else:
    ddpm.summary()
    print("no action")


if __name__=="__main__":
    main()
