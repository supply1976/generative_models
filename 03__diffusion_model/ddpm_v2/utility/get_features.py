import os, sys, argparse
import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.linalg import sqrtm
from scipy import fftpack
import itertools
import tensorflow as tf
from tensorflow import keras


_MODEL_FILE_WITH_TOP = "/remote/us01home28/richwu/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
_MODEL_FILE_NO_TOP = "/remote/us01home28/richwu/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


def load_inception_model(img_size=299):
  # input range (-1, 1)
  model = keras.Sequential(
    [ 
      keras.Input(shape=(None, None, 3)),
      keras.layers.Resizing(img_size, img_size),
      tf.keras.applications.InceptionV3(
        include_top = False, 
        weights = _MODEL_FILE_NO_TOP,
        input_shape = (img_size, img_size, 3),
        pooling = 'avg',
        ),
    ]
  )
  return model


def load_npz_data(npzfile, rescale_to_01=True):
  images = np.load(npzfile)['images']
  print("original images range", images.min(), images.max(), images.shape)
  if rescale_to_01:
    # if input image range is (-1, 1), scale back to (0, 1)
    images = np.clip(images, -1, 1)
    images = 0.5*(images+1)
    images = np.clip(images, 0, 1)
  print("rescale to", images.min(), images.max())
  #n, h, w, c = images.shape
  assert len(images.shape) == 4
  return images


class PatternMetrics:
  def __init__(self, images, pitch=None):
    # images range by default is (0, 1)
    self.images = images
    if pitch is None:
      self.pitch = 1
    else:
      self.pitch = pitch
    self.nums, self.h, self.w, self.c = images.shape
    assert self.h==self.w
    
  def calc_grad_img_SE2D(self, channel_id=0):
    # Shannon entropy of gradient image
    print("calculating gradient image SE 2D")
    images = self.images[:, :, :, channel_id]
    # convert to 4-bit image (0 ~ 15)
    images = (images*15).astype(np.uint8)
    fy, fx = np.gradient(images, axis=(1,2))
    assert len(fx.shape)==3
    assert len(fy.shape)==3
    # remove edge
    fx = fx[:, 1:-1, 1:-1]
    fy = fy[:, 1:-1, 1:-1]
    # histogram counts
    pdf_fx = np.array([np.histogram(_, bins=31)[0] for _ in fx])
    pdf_fy = np.array([np.histogram(_, bins=31)[0] for _ in fy])
    # calc Shannon Entropy
    H_fx = sps.entropy(pk=pdf_fx, axis=1)
    H_fy = sps.entropy(pk=pdf_fy, axis=1)
    features = np.stack([H_fx, H_fy], axis=-1)
    return features

  def calc_PSD(self, channel_id=0, fft_pad=0, remove_dc=True):
    """
    Calculate power spectrum density (PSD) using fft2 (by definition)
    """
    print("calculating PSD")
    images = self.images[:,:,:, channel_id]
    print("images range", images.max(), images.min())
    n = self.w + fft_pad
    self.freqs = fftpack.fftfreq(n, d=self.pitch)
    self.freqs = fftpack.fftshift(self.freqs)
    z = fftpack.fft2(images, shape=(n,n))
    #dc = np.real(z[:,0,0])
    # remove dc
    if remove_dc:
      z[:,0,0]=1.0e-6
    z = fftpack.fftshift(z, axes=(-2,-1))
    psd = np.abs(z)
    self.kx, self.ky = np.meshgrid(self.freqs, self.freqs)
    # normalization
    psd = psd / np.sum(psd)
    return psd
    
  def calc_psd_SE2D(self, psd):
    """
    Calculate 2D Shannon Entropy in freq. domain, NOT in image space domain
    using scipy.stats.entropy()
    Concept:
    For a given pattern clip (rasterized grey scale image), treat the freq spectrum
    as some kind of probability distrubiton of two random variables (X, Y)
    then calculate the following metrics
    S0:    joint entropy H(X,Y)
    S0x:   single variable entropy H(X)
    S0y:   single varaible entropy H(Y)
    Hxy:  conditional entropy H(X|Y) === H(X,Y) - H(Y)
    Hyx:  conditional entropy H(Y|X) === H(X,Y) - H(X)
    Ixy:   mutual information I(x;y) = I(y;x) === H(X) + H(Y) - H(X,Y)
    Note: only 3 of these 6 metrics are indenpendt,
    if a pattern can be fully XY-decopuled, then Ixy=0, otherwise Ixy>0
    """
    print("calculating PSD SE2D")
    nums, nx, ny = psd.shape
    assert nx==ny
    psd_flat = psd.reshape(nums, -1)
    psdx = psd.sum(axis=1)
    psdy = psd.sum(axis=2)
    #psdx = (self.kx**2) * psd
    #psdy = (self.ky**2) * psd
    #psdx = psdx.reshape(nums, -1)
    #psdy = psdy.reshape(nums, -1)
    S0 = sps.entropy(pk=psd_flat, axis=1)
    S0x = sps.entropy(pk=psdx, axis=1)
    S0y = sps.entropy(pk=psdy, axis=1)
    Ixy = S0x + S0y - S0
    Hxy = S0 - S0y
    Hyx = S0 - S0x
    #features = np.stack([Hxy, Hyx, Ixy], axis=-1)
    features = np.stack([S0, S0x, S0y], axis=-1)
    return features

  def calc_inceptionV3_features(self, model):
    print("calculating inceptionV3 model features")
    n, h, w, c = self.images.shape
    # for inceptionV3 model, the input image range is (-1, 1)
    images = 2*self.images -1  ; # (0, 1) -> (-1,1)
    if c < 3:
      images = np.concatenate([images, np.zeros([n, h, w, 3-c])], axis=-1)
    features = model.predict(images, batch_size=32, verbose=1)
    n, fdim = features.shape
    assert fdim==2048
    return features


def save_features(PM, inceptionV3_model, save_dir, output_fn, save_psd):
  grad_img_SE2D_feats = PM.calc_grad_img_SE2D()
  psd = PM.calc_PSD()
  psd_SE2D_feats = PM.calc_psd_SE2D(psd)
  inceptV3_feats = PM.calc_inceptionV3_features(inceptionV3_model)

  df1 = pd.DataFrame(grad_img_SE2D_feats, columns=['Hx', 'Hy'])
  df2 = pd.DataFrame(psd_SE2D_feats, columns=['S0', 'S0x', 'S0y'])
  df3 = pd.DataFrame(inceptV3_feats, 
    columns=["inceptV3_"+str(i) for i in range(2048)])
  df = pd.concat([df1, df2, df3], axis=1)
  df.to_csv(os.path.join(save_dir, output_fn), sep='\t', na_rep="NA", index=False)
  print(os.path.join(save_dir, output_fn), "saved")
  print("PSD shape", psd.shape)
  if save_psd:
    np.savez_compressed(os.path.join(save_dir, "psd"), psd=psd)
  return df


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--real_npz', type=str, default=None)
  parser.add_argument('--workdir', type=str, default=None)
  parser.add_argument('--pitch', type=float, default=None)
  parser.add_argument('--save_psd', action='store_true')
  FLAGS, _ = parser.parse_known_args()
  
  inceptionV3_model = load_inception_model()
  
  if FLAGS.real_npz is not None:
    # processing on real data
    real_data_path = os.path.abspath(FLAGS.real_npz)
    save_dir = os.path.dirname(real_data_path)
    real_images = load_npz_data(real_data_path, rescale_to_01=False)
    _, h, w, _ = real_images.shape 
    PM_real = PatternMetrics(real_images, pitch=FLAGS.pitch)
    df_feats_real = save_features(
      PM_real, inceptionV3_model, save_dir, "real_features.csv", save_psd=FLAGS.save_psd)
    print(df_feats_real.head())

  if FLAGS.workdir is not None:
    assert os.path.isdir(FLAGS.workdir)
    gen_npzs = []
    for root, dirs, files in os.walk(FLAGS.workdir):
      for fn in files:
        if fn.endswith(".npz") and not fn=="psd.npz":
          gen_npzs.append(os.path.join(root, fn))
    if len(gen_npzs)==0:
      return
    for i, npz in enumerate(gen_npzs):
      print(i, npz)
      save_dir = os.path.dirname(npz)
      f, ext = os.path.splitext(os.path.basename(npz))
      gen_images = load_npz_data(npz, rescale_to_01=True)
      PM_gen = PatternMetrics(gen_images, pitch=FLAGS.pitch)
      df_feats_gen = save_features(
        PM_gen, inceptionV3_model, save_dir, f+"_features.csv", save_psd=FLAGS.save_psd)
      print(df_feats_gen.head())
        


if __name__=="__main__":
  main()
