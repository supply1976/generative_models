import os
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from dtype_util import get_compute_dtype

logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(
        self,
        data_dir,
        img_size,
        crop_size=None,
        npz_key='image',
        file_format='.npz',
        dataset_repeat=1,
        train_split=0.8,
        shuffle_buffer=10000,
    ):
        """
        DataLoader for loading and preprocessing .npz image datasets.

        Args:
            data_dir (str or Path): Directory containing .npz files.
            img_size (int): Target image size.
            crop_size (int, optional): Center crop size. Defaults to None.
            npz_key (str): Key for image array in .npz files.
            file_format (str): File extension to search for.
            dataset_repeat (int): Number of times to repeat the dataset.
            train_split (float): Proportion of data for training.
            shuffle_buffer (int): Buffer size for shuffling.
        """
        self.CLIP_MAX = 1.0
        self.CLIP_MIN = -1.0
        self.img_size = img_size
        self.crop_size = crop_size
        self.npz_key = npz_key
        self.data_dir = Path(data_dir).resolve()
        self.dataset_repeat = dataset_repeat
        self.train_split = train_split
        self.shuffle_buffer = shuffle_buffer
        self.file_format = file_format

        assert self.data_dir.exists(), f"data_dir {self.data_dir} does not exist"
        assert self.data_dir.is_dir(), f"data_dir {self.data_dir} is not a directory"
        assert self.npz_key, "npz_key should not be None"
        assert self.file_format, "file_format should not be None"

        self.train_npzfiles, self.valid_npzfiles = self._split_files()
        logging.info(f"Found {len(self.train_npzfiles)} training files")
        logging.info(f"Found {len(self.valid_npzfiles)} validation files.")

        self.train_ds = tf.data.Dataset.from_tensor_slices([str(f) for f in self.train_npzfiles])
        self.valid_ds = tf.data.Dataset.from_tensor_slices([str(f) for f in self.valid_npzfiles])

    def _split_files(self):
        """Split files into train and validation sets."""
        npz_files = list(self.data_dir.rglob(f'*{self.file_format}'))
        assert npz_files, f"No {self.file_format} files found in {self.data_dir}"
        np.random.shuffle(npz_files)
        split_idx = int(len(npz_files) * self.train_split)
        train_files = npz_files[:split_idx]
        valid_files = npz_files[split_idx:]
        assert train_files, "No training files found"
        assert valid_files, "No validation files found"
        return train_files, valid_files

    @staticmethod
    def _preprocess(x, npz_key, crop_size, img_size, clip_min, clip_max):
        try:
            raw_name = x.decode('utf-8')
            data = np.load(raw_name)
            assert npz_key in data, f"Key '{npz_key}' not found in {raw_name}"
            arr = data[npz_key].astype(np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            h, w, c = arr.shape
            if crop_size is not None:
                llx = (h - crop_size) // 2
                lly = (w - crop_size) // 2
                urx = llx + crop_size
                ury = lly + crop_size
                arr = arr[llx:urx, lly:ury, :]
            arr = arr * (clip_max - clip_min) + clip_min
            return arr
        except Exception as e:
            logging.error(f"Error processing {x}: {e}")
            raise

    def _load_npz(self, path):
        def _wrapped_preprocess(x):
            return DataLoader._preprocess(
                x,
                self.npz_key,
                self.crop_size,
                self.img_size,
                self.CLIP_MIN,
                self.CLIP_MAX,
            )
        img = tf.numpy_function(_wrapped_preprocess, [path], get_compute_dtype())
        img_size = self.img_size if self.crop_size is None else self.crop_size
        img = tf.ensure_shape(img, [img_size, img_size, None])
        return img

    def get_dataset(self):
        """
        Returns:
            tuple: (train_ds, valid_ds) as tf.data.Dataset objects.
        """
        train_ds = (
            self.train_ds
            .cache()
            .repeat(self.dataset_repeat)
            .shuffle(buffer_size=self.shuffle_buffer)
            .map(self._load_npz, num_parallel_calls=tf.data.AUTOTUNE)
        )
        valid_ds = (
            self.valid_ds
            .cache()
            .map(self._load_npz, num_parallel_calls=tf.data.AUTOTUNE)
        )
        return train_ds, valid_ds

def unit_test():
    # Example unit test for DataLoader
    import tempfile
    import numpy as np

    # Create a temporary directory with fake .npz files
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(10):
            arr = np.random.rand(32, 32).astype(np.float32)
            np.savez(os.path.join(tmpdir, f"img_{i}.npz"), image=arr)
        loader = DataLoader(tmpdir, img_size=32)
        train_ds, valid_ds = loader.get_dataset()
        for batch in train_ds.take(1):
            print("Train batch shape:", batch.shape)
        for batch in valid_ds.take(1):
            print("Valid batch shape:", batch.shape)

if __name__ == "__main__":
    unit_test()
