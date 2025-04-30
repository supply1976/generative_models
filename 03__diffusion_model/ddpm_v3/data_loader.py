import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from functools import partial
import tensorflow as tf
from tensorflow import keras


class DataLoader:
  def __init__(self, 
    data_dir, 
    crop_size=None, 
    npz_key='image', 
    file_format='.npz',
    dataset_repeat=1,
    ):
    """
    search the numpy (.npz) files in the fiven data_dir
    load numpy data, 
    do data pre-processing
    
    return: tf dataset
    
    """
    self.CLIP_MAX = 1.0
    self.CLIP_MIN = -1.0
    self.crop_size = crop_size
    self.npz_key = npz_key
    self.data_dir = os.path.abspath(data_dir)
    self.dataset_repeat = dataset_repeat
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

  def load_dataset(self):
    train_ds = tf.data.Dataset.from_tensor_slices(self.all_npzfiles)
    valid_ds = tf.data.Dataset.from_tensor_slices(self.valid_npzfiles)

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
      img = (img) *(self.CLIP_MAX-self.CLIP_MIN) + self.CLIP_MIN
      return img

    # train ds
    train_ds = train_ds.cache().repeat(self.dataset_repeat)
    train_ds = train_ds.shuffle(train_ds.cardinality())
    train_ds = train_ds.map(
      lambda x: tf.py_function(_preprocess, [x], tf.float32),
      num_parallel_calls=tf.data.AUTOTUNE)
    # valid ds  
    valid_ds = valid_ds.cache()
    valid_ds = valid_ds.map(
      lambda x: tf.py_function(_preprocess, [x], tf.float32),
      num_parallel_calls=tf.data.AUTOTUNE)

    return (train_ds, valid_ds)


def unit_test():
  # test the DataLoader

  pass 
  #data_dir = sys.argv[1]
  #dataloader = DataLoader(data_dir=data_dir)
  #ds = dataloader.load_dataset(batch_size=3)
  #for x in ds.take(1):
  #  print(x.shape, type(x), tf.shape(x))


if __name__ == "__main__":
  unit_test()
