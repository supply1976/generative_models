import os, sys, argparse
import gzip
import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.linalg import sqrtm
from scipy import fftpack
from skimage.metrics import structural_similarity as ssim
import itertools
import tensorflow as tf
from tensorflow import keras


_MODEL_FILE_WITH_TOP = "/remote/us01home28/richwu/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
_MODEL_FILE_NO_TOP = "/remote/us01home28/richwu/Downloads/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


def load_inception_model(img_size=299):
  # input range need to be (-1, 1)
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


def lempel_ziv_complexity_fast(binary_sequence):
  """
  Fast Lempel-Ziv Complexity for a 1D binary sequence.
  Input: binary_sequence (1D numpy array or list of 0s and 1s)
  Output: LZC (int)
  """
  s = ''.join(str(int(b)) for b in binary_sequence)
  n = len(s)
  i, lzc = 0, 1
  substrings = set()
  k = 1
  while i + k <= n:
    substr = s[i:i+k]
    if substr not in substrings:
      lzc += 1
      substrings.add(substr)
      i += k
      k = 1
    else:
      k += 1
      if i + k > n:
        break
  return lzc


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


  def calc_grad_img_entropy(self, nbits=1, channel_id=0):
    # Shannon entropy of gradient image
    print("calculating gradient image Shannon entropy")
    images = self.images[:, :, :, channel_id]
    assert nbits >= 1
    max_val = 2**nbits -1
    if nbits==1:
      images = (images > 0.5).astype(np.uint8)
    else:
      images = (images * max_val).astype(np.uint8)
    # calc gradient images
    fy, fx = np.gradient(images, axis=(1,2))
    assert len(fx.shape)==3
    assert len(fy.shape)==3
    # remove edge
    fx = fx[:, 1:-1, 1:-1]
    fy = fy[:, 1:-1, 1:-1]
    # histogram counts
    fx_hists, fy_hists = ([], [])
    for i in range(fx.shape[0]):
      fx_val, fx_count = np.unique(fx[i], return_counts=True)
      fy_val, fy_count = np.unique(fy[i], return_counts=True)
      assert len(fx_count) <= (max_val*2+1)
      assert len(fy_count) <= (max_val*2+1)
      # append 0 to make the total length equal to max_val*2+1
      if len(fx_count)<max_val*2+1:
        fx_count = np.insert(fx_count, 0, [0]*(max_val*2+1 - len(fx_count)))
      if len(fy_count)<max_val*2+1:
        fy_count = np.insert(fy_count, 0, [0]*(max_val*2+1 - len(fy_count)))
      #
      fx_hists.append(fx_count)
      fy_hists.append(fy_count)
    # get all gradient image histograms
    fx_hists = np.stack(fx_hists, axis=0)
    fy_hists = np.stack(fy_hists, axis=0)
    print(fx_hists.shape)
    # calc Shannon Entropy
    H_fx = sps.entropy(pk=fx_hists, axis=1)
    H_fy = sps.entropy(pk=fy_hists, axis=1)

    features = np.stack([H_fx, H_fy], axis=-1)
    return features


  def calc_PSD(self, channel_id=0, fft_pad=0):
    """
    Calculate power spectrum density (PSD) using fft2 (by definition)
    """
    print("calculating PSD")
    images = self.images[:,:,:, channel_id]
    # normalize to volumn = 0 (remove dc)
    images = images - images.mean(axis=(1,2))[:, None, None]
    print("images range, remove dc", images.max(), images.min(), images.mean())

    n = self.w + fft_pad
    self.freqs = fftpack.fftfreq(n, d=self.pitch)
    self.freqs = fftpack.fftshift(self.freqs)
    z = fftpack.fft2(images, shape=(n,n))
    z = fftpack.fftshift(z, axes=(1,2))
    psd = np.abs(z)**2
    self.kx, self.ky = np.meshgrid(self.freqs, self.freqs)
    # normalization
    return psd


  def calc_fftPSD_SE2D(self, psd):
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


  def calc_LZCxy_Pden_gzipSize(self, channel_id=0):
    images = self.images[:, :, :, channel_id]
    pdens = images.mean(axis=(1,2))
    # use binary images to calculate LZC
    images_bin = (self.images[:, :, :, channel_id] > 0.5).astype(np.uint8)
    lzc_x_vals, lzc_y_vals, comp_sizes = ([], [], [])
    for img in images_bin:
      lzc_x = lempel_ziv_complexity_fast(img.flatten())
      lzc_y = lempel_ziv_complexity_fast(img.T.flatten())
      compressed_size = len(gzip.compress(img.flatten()))
      lzc_x_vals.append(lzc_x)
      lzc_y_vals.append(lzc_y)
      comp_sizes.append(compressed_size)
    features = np.stack([lzc_x_vals, lzc_y_vals, pdens, comp_sizes], axis=-1)
    return features


def save_features(PM, channel_id, inceptionV3_model, save_dir, output_fn):
  feats_grad_img_H = PM.calc_grad_img_entropy(nbits=1, channel_id=channel_id)
  psd = PM.calc_PSD(channel_id=channel_id)
  feats_fftPSD_SE2D = PM.calc_fftPSD_SE2D(psd)
  feats_inceptV3 = PM.calc_inceptionV3_features(inceptionV3_model)
  feats_LZC_Pden_gzipSize = PM.calc_LZCxy_Pden_gzipSize(channel_id=channel_id)

  
  df1 = pd.DataFrame(feats_grad_img_H, columns=['Hx', 'Hy'])
  df2 = pd.DataFrame(feats_fftPSD_SE2D, columns=['S0', 'S0x', 'S0y'])
  df3 = pd.DataFrame(feats_LZC_Pden_gzipSize, 
    columns=['LZC_x', 'LZC_y','Pden', 'gzipSize'])

  df4 = pd.DataFrame(feats_inceptV3, 
    columns=["inceptV3_"+str(i) for i in range(2048)])
  
  df = pd.concat([df1, df2, df3, df4], axis=1)
  df.to_csv(os.path.join(save_dir, output_fn), sep='\t', na_rep="NA", index=False)
  print(os.path.join(save_dir, output_fn), "saved")
  return df


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--real_npz', type=str, default=None)
  parser.add_argument('--workdir', type=str, default=None)
  parser.add_argument('--pitch', type=float, default=None)
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
      PM=PM_real, 
      channel_id=0,
      inceptionV3_model=inceptionV3_model, 
      save_dir=save_dir, 
      output_fn="real_features_"+str(h)+"x"+str(w)+".csv")
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
      gen_images = load_npz_data(npz, rescale_to_01=False)
      PM_gen = PatternMetrics(gen_images, pitch=FLAGS.pitch)
      df_feats_gen = save_features(
        PM=PM_gen,
        channel_id=0,
        inceptionV3_model=inceptionV3_model,
        save_dir=save_dir,
        output_fn=f+"_features.csv")
      print(df_feats_gen.head())
        


if __name__=="__main__":
  main()
