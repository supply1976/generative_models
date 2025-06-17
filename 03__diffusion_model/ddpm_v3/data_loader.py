import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from functools import partial
import tensorflow as tf
from tensorflow import keras


class DataLoader:
  def __init__(self, 
    data_dir,
    img_size,
    crop_size=None, 
    npz_key='image', 
    file_format='.npz',
    ):
    """
    search the numpy (.npz) files in the fiven data_dir
    load numpy data, 
    do data pre-processing
    
    return: tf dataset
    
    """
    self.CLIP_MAX = 1.0
    self.CLIP_MIN = -1.0
    self.img_size = img_size
    self.crop_size = crop_size
    self.npz_key = npz_key
    self.data_dir = os.path.abspath(data_dir)
    assert os.path.exists(self.data_dir), f"data_dir {self.data_dir} not exists"
    assert os.path.isdir(self.data_dir), f"data_dir {self.data_dir} is not a directory"
    assert self.npz_key is not None, "npz_key should not be None"
    assert file_format is not None, "file_format should not be None"

    self.train_npzfiles = []
    self.valid_npzfiles = []
    # search the npz files in the data_dir
    for root, dirs, files in os.walk(self.data_dir):
      for fn in files:
        if fn.endswith(file_format):
          file_path = os.path.join(root, fn)
          if np.random.rand() < 0.8:  # 80% for training, 20% for validation
            self.train_npzfiles.append(file_path)
          else:
            self.valid_npzfiles.append(file_path)
    assert len(self.train_npzfiles) > 0
    assert len(self.valid_npzfiles) > 0
    print(f"Found {len(self.train_npzfiles)} training files")
    print(f"Found {len(self.valid_npzfiles)} validation files.")
    self.all_npzfiles = self.train_npzfiles + self.valid_npzfiles
    self.train_ds = tf.data.Dataset.from_tensor_slices(self.all_npzfiles)
    self.valid_ds = tf.data.Dataset.from_tensor_slices(self.valid_npzfiles)
    #print(self.train_ds)
    #print(self.valid_ds)

  def _load_npz(self, path):

    def _preprocess(x):
      #raw_name = x.numpy().decode()
      raw_name = x.decode('utf-8')
      data = np.load(raw_name)
      assert self.npz_key in list(data.keys())
      arr = data[self.npz_key].astype(np.float32)
      if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=-1)
      self.h, self.w, self.c = arr.shape

      if self.crop_size is not None:
        h, w, _ = arr.shape
        llx = (h - self.crop_size) // 2
        lly = (w - self.crop_size) // 2
        urx = llx + self.crop_size
        ury = lly + self.crop_size
        arr = arr[llx:urx, lly:ury, :]
        
      arr = (arr) *(self.CLIP_MAX-self.CLIP_MIN) + self.CLIP_MIN
      return arr
    img = tf.numpy_function(_preprocess, [path], tf.float32)
    img_size = self.img_size if self.crop_size is None else self.crop_size
    img = tf.ensure_shape(img, [img_size, img_size, None])
    
    return img
    
  def _get_dataset(self):
    # train ds
    train_ds = (
      self.train_ds
      .map(self._load_npz, num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
      .shuffle(buffer_size=10000)
      .repeat()
    )
    # valid ds  
    valid_ds = (
      self.valid_ds
      .map(self._load_npz, num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
    )
    return (train_ds, valid_ds)


def unit_test():
  # test the DataLoader
  pass 


if __name__ == "__main__":
  unit_test()
