import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from functools import partial
import tensorflow as tf
from tensorflow import keras


(CLIP_MIN, CLIP_MAX) = (-1.0, 1.0)


class DataLoader:
  def __init__(self, data_dir, crop_size=None, npz_key='image', file_format='.npz'):
    """
    search the numpy (.npz) files in the fiven data_dir
    load numpy data, 
    do data pre-processing
    
    return: tf dataset
    
    """
    self.crop_size = crop_size
    self.npz_key = npz_key
    self.data_dir = os.path.abspath(data_dir)
    self.npz_files = []
    for root, dirs, files in os.walk(self.data_dir):
      for fn in files:
        if fn.endswith(file_format):
          file_path = os.path.join(root, fn)
          self.npz_files.append(file_path)
  
  def load_dataset(self):
    if len(self.npz_files) == 0:
      return None
    ds = tf.data.Dataset.from_tensor_slices(self.npz_files)

    def _preprocess(x):
      raw_name = x.numpy().decode()
      data = np.load(raw_name)
      assert self.npz_key in list(data.keys())
      img = data[self.npz_key].astype(np.float32)
      if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
      h, w, c = img.shape
      assert h==w

      if self.crop_size is not None:
        llx = (h - self.crop_size)//2
        lly = (w - self.crop_size)//2
        urx = llx + self.crop_size
        ury = lly + self.crop_size
        img = img[llx:urx, lly:ury, :]
      img = (img) *(CLIP_MAX-CLIP_MIN) + CLIP_MIN
      return img

    ds = ds.cache()
    ds = ds.shuffle(ds.cardinality())
    ds = ds.map(lambda x: tf.py_function(_preprocess, [x], tf.float32),
      num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def unit_test():
  data_dir = sys.argv[1]
  dataloader = DataLoader(data_dir=data_dir)
  ds = dataloader.load_dataset(batch_size=3)
  for x in ds.take(1):
    print(x.shape, type(x), tf.shape(x))


if __name__ == "__main__":
  unit_test()
