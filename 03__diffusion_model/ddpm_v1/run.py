import os, sys, argparse
import math, time, datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import modelDef
from data_loader import DataLoader


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--training',      action='store_true')
  parser.add_argument('--gen_images',    action='store_true')
  parser.add_argument('--num_gen_images',type=int, default=100)
  parser.add_argument('--restore_model', type=str, default=None)
  parser.add_argument('--dataset_path',  type=str, default=None,
    help=" data dir with .npz files")
  parser.add_argument('--dataset_name',  type=str, default=None)
  parser.add_argument('--given_samples', type=str, default=None)
  # model configs
  parser.add_argument('--img_size',        type=int, required=True)
  parser.add_argument('--img_channel',     type=int, default=1)
  parser.add_argument('--epochs',          type=int, default=20)
  parser.add_argument('--batch_size',      type=int, default=4)
  parser.add_argument('--beta_schedule',   type=str, default='linear')
  parser.add_argument('--beta_start',      type=float, default=1.0e-3)
  parser.add_argument('--beta_end',        type=float, default=0.2)
  parser.add_argument('--total_timesteps', type=int, default=1000)
  parser.add_argument('--num_res_blocks', type=int, default=2)
  parser.add_argument('--norm_groups',     type=int, default=8, 
    help="number of groups in group_normalization()")
  parser.add_argument('--learning_rate',   type=float, default=0.0001)
  parser.add_argument('--first_ch',        type=int, default=8,
    help="first convolution channel")
  parser.add_argument('--ch_mul',nargs='+',type=int, default=[1, 2, 4, 8],
    help="channel multiplier")
  parser.add_argument('--OUTPUT_PATH', type=str, default = "training_outputs")

  FLAGS, _ = parser.parse_known_args()
  if not os.path.isdir(FLAGS.OUTPUT_PATH): os.mkdir(FLAGS.OUTPUT_PATH)

  # model configs
  img_size        = FLAGS.img_size
  img_channel     = FLAGS.img_channel
  batch_size      = FLAGS.batch_size
  epochs          = FLAGS.epochs
  total_timesteps = FLAGS.total_timesteps
  beta_schedule   = FLAGS.beta_schedule
  beta_start      = FLAGS.beta_start
  beta_end        = FLAGS.beta_end
  norm_groups     = FLAGS.norm_groups  
  learning_rate   = FLAGS.learning_rate
  first_ch        = FLAGS.first_ch
  ch_mul          = FLAGS.ch_mul
  num_res_blocks  = FLAGS.num_res_blocks  # Number of residual blocks
  has_attention   = [False, False, True, False]
  assert len(ch_mul)==len(has_attention)
  widths = [first_ch * mult for mult in ch_mul]
    
  # GPU devices
  gpus = tf.config.list_physical_devices("GPU")
  #logical_gpus = tf.config.list_logical_devices('GPU')
  #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  if gpus: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

  # Build the unet model
  network = modelDef.build_model(
    img_size=img_size, 
    img_channel=img_channel,
    first_conv_channels=first_ch,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish)

  ema_network = modelDef.build_model(
    img_size=img_size, 
    img_channel=img_channel,
    first_conv_channels=first_ch,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish)

  network.summary()
  # Initially the weights are the same
  ema_network.set_weights(network.get_weights())  

  # Get an instance of the Gaussian Diffusion utilities
  gausdiff_util = modelDef.GausDiffUtil(
    beta_start=beta_start, 
    beta_end=beta_end, 
    beta_schedule=beta_schedule, timesteps=total_timesteps)
   
  # restore the trained model weights
  if FLAGS.restore_model is not None:
    model_path = os.path.abspath(FLAGS.restore_model)
    model_dir = os.path.dirname(model_path)
    print("restoring model from: {}".format(model_path))
    load_status = ema_network.load_weights(model_path)
    network.set_weights(ema_network.get_weights())
    #load_status.assert_consumed()

  # Get the diffusion model
  ddpm = modelDef.DiffusionModel(
    network=network, 
    ema_network=ema_network,
    gausdiff_util=gausdiff_util,
    timesteps=total_timesteps)

  train_ds = None
  if FLAGS.dataset_name is None:
    print("no training dataset is provided, skip training")

  elif FLAGS.dataset_name in ["cifar10", "mnist"]:
    # use internal keras dataset for quick testing
    print(" >> use internal keras dataset {} as training data".format(FLAGS.dataset_name))
    if FLAGS.dataset_name == "cifar10":
      (train_images, train_labels), _ = keras.datasets.cifar10.load_data()
      carID, _ = np.where(train_labels==1)
      train_images = train_images[carID]

    elif FLAGS.dataset_name == "mnist": 
      (train_images, _), _ = keras.datasets.mnist.load_data()
      train_images = np.expand_dims(train_images, axis=-1)

    else:
      train_images = None
    train_images = train_images.astype(np.float32)
    train_images = 2*(train_images / 255.0) - 1 ; # rescale to (-1, 1)
    if FLAGS.dataset_name=="mnist":
      train_images = tf.image.resize(train_images, [32, 32])
    #
    train_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=10000)

  else:
    if FLAGS.dataset_path is None:
      print("please provide dataset_path, Exit")
      return
    else:
      print("user defined dataset name:{}".format(FLAGS.dataset_name))
      if os.path.isdir(FLAGS.dataset_path):
        # a folder contains multiple npz files
        dataloader = DataLoader(data_dir=FLAGS.dataset_path, crop_size=None)
        train_ds = dataloader.load_dataset()
      elif os.path.isfile(FLAGS.dataset_path):
        # single (big) npz file
        data = np.load(FLAGS.dataset_path)
        train_images = data['images']
        train_images = 2.0 * train_images - 1.0
        train_ds = tf.data.Dataset.from_tensor_slices(train_images)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(buffer_size=10000)
      else:
        return

  if FLAGS.training and train_ds is not None:
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    # get input image shape
    for x in train_ds.take(1):
      _, h, w, c = x.shape
      print(x.shape, type(x))
    assert c==img_channel
    assert h==img_size
    
    dateID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    savedir = "DDPM_{}x{}x{}__dataset_{}".format(h, w, c, FLAGS.dataset_name)
    savedir = os.path.join(FLAGS.OUTPUT_PATH, savedir)
    if not os.path.isdir(savedir): os.mkdir(savedir)
    savedir = os.path.join(savedir, dateID)
    os.mkdir(savedir)

    callback_save_model = keras.callbacks.LambdaCallback(
      on_epoch_end=lambda epoch,logs: ddpm.save_model(epoch, savedir=savedir))
    callback_genimages = keras.callbacks.LambdaCallback(
      on_train_end=ddpm.generate_images)

    callback_list = [callback_save_model, callback_genimages]

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
    #ema_network.save_weights(os.path.join(savedir, "ema_best.weights.h5"))
    
  elif FLAGS.gen_images:
    #savedir = model_dir 
    gen_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gen_dir = os.path.join(model_dir, "gen_"+gen_date)
    os.mkdir(gen_dir)

    if FLAGS.given_samples is None:
      # use gaussian random noise to generate images
      ddpm.generate_images(
        num_images=FLAGS.num_gen_images, savedir=gen_dir, export_intermediate=False)
    else:
      # use external given images (.npz) as input to generate images
      data = np.load(FLAGS.given_samples)
      data = tf.convert_to_tensor(data['images'], tf.float32)
      ddpm.generate_images(
        given_samples=data, savedir=gen_dir, freeze_1st=True, export_intermediate=False)

  else:
    ddpm.summary()
    print("no action")


if __name__=="__main__":
    main()
