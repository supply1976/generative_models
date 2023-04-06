import os, sys
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()


class VariationalAutoEncoder:
  def __init__(self, latent_dim:int, inp_shape):
    self.latent_dim = latent_dim
    self.inp_shape = inp_shape
    self.x = keras.Input(shape=inp_shape, name="encoder_input")
    self.z = keras.Input(shape=(latent_dim,), name="decoder_input")

  def _encoding(self, _units=[]):
    # _units: list of int, can be empty []
    # return: encoder model using Dense layers
    outputs = []
    _x = keras.layers.Flatten(name="flatten")(self.x)
    outputs.append(_x)
    if len(_units)==0:
      pass
    else:
      for i, u in enumerate(_units):
        h = keras.layers.Dense(units=u, activation='relu', name="enc"+str(i+1))
        _y = h(outputs[-1])
        outputs.append(_y)
    z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(outputs[-1])
    z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(outputs[-1])
    #z_log_var = tf.zeros_like(z_mean)
    rnv = tf.random.normal(shape=tf.shape(z_mean))
    z_noise = z_mean + tf.exp(0.5*z_log_var) * rnv
    model = keras.Model(
      inputs=self.x, 
      outputs=[z_mean, z_log_var, z_noise], 
      name="encoder_model")
    return model

  def _decoding(self, _units=[]):
    # _units: list of int, can be empty []
    # return: decoder model
    outputs = []
    outputs.append(self.z)
    _units.append(np.prod(self.inp_shape))
    for i, u in enumerate(_units):
      actf = 'sigmoid' if i==len(_units)-1 else 'relu'
      h = keras.layers.Dense(units=u, activation=actf, name="dec"+str(i+1))
      _y = h(outputs[-1])
      outputs.append(_y)
    _y_final = keras.layers.Reshape(self.inp_shape, name="reshape")(outputs[-1])
    model = keras.Model(inputs=self.z, outputs=_y_final, name="decoder_model")
    return model

def view_images(imgs, ncols=10):
  if len(imgs)>100:
    imgs = imgs[0:100]
  nrows = len(imgs)//ncols if len(imgs)%ncols==0 else 1+len(imgs)//ncols
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,4))
  axes = axes.flatten()
  for i, img in enumerate(imgs):
    axes[i].imshow(img)
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)


def view_encoded_result(encoded_vec, labels):
  _vec_norm = np.linalg.norm(encoded_vec, axis=-1)
  _, vec_dim = encoded_vec.shape
  fig = plt.figure(figsize=(10,3))
  ax1 = fig.add_subplot(121)
  proj = '3d' if vec_dim==3 else None
  ax2 = fig.add_subplot(122, projection=proj)
  for i in range(10):
    vec_i = encoded_vec[np.where(labels==i)]
    if vec_dim==2:
      x, y = vec_i.T
      ax2.plot(x, y, '.', label=str(i))
    elif vec_dim==3:
      x, y, z = vec_i.T
      ax2.scatter(x, y, z) 
    else:
      return None
  ax1.hist(_vec_norm, bins=20)
  
def mnist_modify(orig_images, orig_labels, color_channel=3):
  _, h, w = orig_images.shape
  data = []
  new_labels = []
  for i in range(10):
    img_i = orig_images[np.where(orig_labels==i)]
    img_i = img_i[0:color_channel*(len(img_i)//color_channel)]
    # reshape to RGB image
    img_i = img_i.reshape([-1, color_channel, h, w])
    img_i = np.transpose(img_i, [0, 2, 3, 1])
    data.append(img_i)
    new_labels.extend([i]*len(img_i))
  mnist_RGB = np.concatenate(data)
  return (mnist_RGB, np.array(new_labels, dtype=np.int8))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--latent_dim', type=int, default=2)
  parser.add_argument('--savedir', type=str, default='./output')
  parser.add_argument('--inference', action='store_true')
  parser.add_argument('--training', action='store_true')
  
  FLAGS, _ = parser.parse_known_args()
  if not os.path.isdir(FLAGS.savedir):
    os.mkdir(FLAGS.savedir)

  # load the original mnist data
  #data = tf.keras.datasets.fashion_mnist.load_data()
  data = tf.keras.datasets.mnist.load_data()
  (train_images, train_labels), (test_images, test_labels) = data
  train_images = train_images.astype(np.float32)/255.0
  test_images = test_images.astype(np.float32)/255.0

  # modify the mnist data to become RGB color image for study purpose
  #train_images, train_labels = mnist_modify(train_images, train_labels)
  #test_images, test_labels = mnist_modify(test_images, test_labels)
  
  inp_shape = train_images.shape[1:]
  vae = VariationalAutoEncoder(FLAGS.latent_dim, inp_shape)
  # build and initialize encoder model
  encoder_model = vae._encoding(_units=[256])
  encoder_model.summary()
  for ly in encoder_model.layers:
    print(ly.name, ly, ly.output.shape)

  # build initialize decoder model
  decoder_model = vae._decoding(_units=[256])
  decoder_model.summary()
  for ly in decoder_model.layers:
    print(ly.name, ly, ly.output.shape)

  # encoder and decoder models are built independent and can be used standalone
  # but they are not yet connected, 
  # now we connect these two nets
  z_mean, z_log_var, z_noise = encoder_model(vae.x)
  x_decoded = decoder_model(z_noise)

  # adding customized loss in latent space (latent vector norm regularization)
  vae_model = keras.Model(inputs=vae.x, outputs=x_decoded, name="VAE")
  mse_loss = tf.reduce_mean(keras.losses.mse(vae.x, x_decoded))
  bce_loss = tf.reduce_mean(keras.losses.binary_crossentropy(vae.x, x_decoded))
  kl_distance = tf.reduce_sum(-z_log_var+tf.square(z_mean)+tf.exp(z_log_var) -1, axis=-1)
  kl_loss = 0.001 * tf.reduce_mean(kl_distance)
  z_mean_L2norm_loss = tf.reduce_mean(tf.square(tf.norm(z_mean, axis=-1) - 1))
  #kl_loss = tf.reduce_mean(keras.losses.mse(tf.zeros_like(kl_distance), kl_distance))
  # add losses
  #vae_model.add_loss(mse_loss)
  #vae_model.add(bce_loss)
  vae_model.add_loss(kl_loss)
  vae_model.add_loss(z_mean_L2norm_loss)
  vae_model.add_metric(mse_loss, name="mse_loss", aggregation='mean')
  vae_model.add_metric(bce_loss, name="bce_loss", aggregation='mean')
  vae_model.add_metric(kl_loss, name="kl_loss", aggregation='mean')
  vae_model.add_metric(z_mean_L2norm_loss, name="z_mean_L2norm_loss", aggregation='mean')
  vae_model.summary()
  vae_model.compile(optimizer='adam', loss='binary_crossentropy')

  # see images before training
  #x_b4tr = autoencoder.predict(test_images[0:10])
  #imgs = np.concatenate([test_images[0:10], x_b4tr])
  #view_images(imgs, ncols=10)

  savedir = os.path.join(FLAGS.savedir, "latent_dim_"+str(FLAGS.latent_dim))
  if not os.path.isdir(savedir):
    os.mkdir(savedir)

  if FLAGS.training:
    vae_model.fit(x=train_images, y=train_images,
      epochs=FLAGS.epochs,
      shuffle=True,
      batch_size=FLAGS.batch_size,
      validation_data=(test_images, test_images))
    vae_model.save(os.path.join(savedir, "best_vae_model.h5"))
    encoder_model.save(os.path.join(savedir, "best_encoder_model.h5"))
    decoder_model.save(os.path.join(savedir, "best_decoder_model.h5"))
    # see images after training
    #x_aftr = autoencoder.predict(test_images[0:10])
    #imgs = np.concatenate([test_images[0:10], x_b4tr, x_aftr])
    #view_images(imgs, ncols=10)
    
  elif FLAGS.inference:
    enc_model_path = os.path.join(savedir, "best_encoder_model.h5")
    dec_model_path = os.path.join(savedir, "best_decoder_model.h5")
    vae_model_path = os.path.join(savedir, "best_vae_model.h5")
    restored_vae_model = keras.models.load_model(vae_model_path)
    restored_encoder_model = keras.models.load_model(enc_model_path)
    restored_decoder_model = keras.models.load_model(dec_model_path)

    #rnd_idx = np.random.choice(len(test_images), 10, replace=False) 
    rnd_idx = range(10)
    test_decoded = restored_vae_model.predict(test_images[rnd_idx])
    imgs = np.concatenate([test_images[rnd_idx], test_decoded])
    view_images(imgs, ncols=10)

    enc_zm, enc_zlogvar, _ = restored_encoder_model.predict(test_images)
    enc_sigma = np.exp(0.5*enc_zlogvar)
    print(np.mean(enc_zm), np.mean(enc_zlogvar))
    view_encoded_result(enc_zm, test_labels)
    view_encoded_result(enc_sigma, test_labels)

    #theta = np.linspace(0, 360, 100)
    #arr = np.array([(np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))) for t in theta])
    #L = np.linspace(0.1, 0.9, 10)
    #test_arr = np.array([[i, j] for i in L for j in L])
    #imgs_gen = decoder_model.predict(arr)
    #view_images(imgs_gen, ncols=10)

  else:
    print("no action")
    return None


if __name__=="__main__":
  main()
  plt.show()


