import os, sys
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt


tf.compat.v1.disable_eager_execution()


class AutoEncoderANN:
    def __init__(self, latent_dim, inp_shape):
        """
        latent_dim: int
        inp_shape: input image shape (height, width)
        """
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
        _units.append(self.latent_dim)
        for i, u in enumerate(_units):
            # do not apply activation on last layer
            actf = None if i==len(_units)-1 else 'relu'
            h = keras.layers.Dense(units=u, activation=actf, name="enc"+str(i+1))
            _y = h(outputs[-1])
            outputs.append(_y)
        # build model graph
        model = keras.Model(inputs=self.x, outputs=outputs[-1], name="encoder_model")
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
        yfinal = keras.layers.Reshape(self.inp_shape, name="reshape")(outputs[-1])
        model = keras.Model(inputs=self.z, outputs=yfinal, name="decoder_model")
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
    parser.add_argument('--dataset', type=str, default=None)  
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--enc_inter_dims', type=int, nargs="*", default=[16])
    parser.add_argument('--dec_inter_dims', type=int, nargs="*", default=[16])
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--savedir', type=str, default='./output')
    parser.add_argument('--modeldir', type=str, default='./')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--add_latent_vec_reg_loss', action='store_true')
  
    FLAGS, _ = parser.parse_known_args()
    if not os.path.isdir(FLAGS.savedir): os.mkdir(FLAGS.savedir)

    tr_inputs, test_inputs = (None, None)
    tr_targets, test_targets = (None, None)
    if FLAGS.dataset is None:
        # load the original mnist data
        # data = tf.keras.datasets.fashion_mnist.load_data()
        data = tf.keras.datasets.mnist.load_data()
        (tr_inputs, tr_targets), (test_inputs, test_targets) = data
        tr_inputs = tr_inputs.astype(np.float32)/255.0
        test_inputs = test_inputs.astype(np.float32)/255.0
        # modify the mnist data to become RGB color image for study purpose
        tr_inputs, tr_targets = mnist_modify(tr_inputs, tr_targets)
        test_inputs, test_targets = mnist_modify(test_inputs, test_targets)
        
    elif FLAGS.dataset.endswith('npy'):
        data = np.load(FLAGS.dataset, allow_pickle='True')
        tr_inputs = data.astype(np.float32)
        test_inputs = tr_inputs
       
    else:
        return 0

    inp_shape = tr_inputs.shape[1:]
    ae_ann = AutoEncoderANN(FLAGS.latent_dim, inp_shape)
    # build and initialize encoder model
    encoder_model = ae_ann._encoding(_units=FLAGS.enc_inter_dims)
    encoder_model.summary()
    for ly in encoder_model.layers:
        print(ly.name, ly, ly.output.shape)

    # build initialize decoder model
    decoder_model = ae_ann._decoding(_units=FLAGS.dec_inter_dims)
    decoder_model.summary()
    for ly in decoder_model.layers:
        print(ly.name, ly, ly.output.shape)

    # encoder and decoder models are built independent and can be used standalone
    # but they are not yet connected, now we connect these two nets
    x_encoded = encoder_model(ae_ann.x)
    x_decoded = decoder_model(x_encoded)
    AE_model = keras.Model(inputs=ae_ann.x, outputs=x_decoded, name="AE")

    # adding customized loss in latent space (latent vector norm regularization)
    latent_vec_norm_loss = tf.reduce_mean(tf.square(tf.norm(x_encoded, axis=-1)-1))
    AE_model.add_metric(
        latent_vec_norm_loss, name='latent_vec_norm_loss', aggregation='mean')
    if FLAGS.add_latent_vec_reg_loss:
        AE_model.add_loss(latent_vec_norm_loss)
    AE_model.summary()
    AE_model.compile(optimizer='adam', loss='mse')

    # see decoded result before training
    x_dec_b4tr = AE_model.predict(test_inputs[0:10])

    savedir = os.path.join(FLAGS.savedir, "latent_dim_"+str(FLAGS.latent_dim))
    if not os.path.isdir(savedir): os.mkdir(savedir)

    if FLAGS.training:
        AE_model.fit(
            x=tr_inputs, 
            y=tr_inputs,
            epochs=FLAGS.epochs,
            shuffle=True,
            batch_size=FLAGS.batch_size,
            validation_data=(test_inputs, test_inputs))
        AE_model.save(os.path.join(savedir, "best_ae_model.h5"))
        encoder_model.save(os.path.join(savedir, "best_encoder_model.h5"))
        decoder_model.save(os.path.join(savedir, "best_decoder_model.h5"))
    
        # see images after training
        x_dec_aftr = AE_model.predict(test_inputs[0:10])
        data = np.concatenate([test_inputs[0:10], x_dec_b4tr, x_dec_aftr])
        view_images(data, ncols=10)
    
    elif FLAGS.inference:
        enc_model_path = os.path.join(FLAGS.modeldir, "best_encoder_model.h5")
        dec_model_path = os.path.join(FLAGS.modeldir, "best_decoder_model.h5")
        ae_model_path = os.path.join(FLAGS.modeldir, "best_ae_model.h5")
        restored_AE_model = keras.models.load_model(ae_model_path)
        restored_encoder_model = keras.models.load_model(enc_model_path)
        restored_decoder_model = keras.models.load_model(dec_model_path)
        rnd_idx = np.random.choice(len(test_inputs), 10, replace=False) 
        test_decoded = restored_AE_model.predict(test_inputs[rnd_idx])
        data = np.concatenate([test_inputs[rnd_idx], test_decoded])
        view_images(data, ncols=10)
        # see the encoded latent vector distribution
        enc_vec = restored_encoder_model.predict(test_inputs)
        if test_targets is not None:
            view_encoded_result(enc_vec, test_targets)

    else:
        print("no action")
        return None


if __name__=="__main__":
  main()
  plt.show()


