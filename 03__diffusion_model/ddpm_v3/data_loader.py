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
    crop_type='center',
    crop_position=None,
    image_key='image',
    label_key=None,
    file_format='.npz',
    augment=False,
    augment_type=None,
    ):
    """
    Enhanced DataLoader for DDPM v3 with flexible cropping options.
    
    Searches for numpy (.npz) files in the given data_dir, loads numpy data,
    and performs comprehensive data preprocessing including various cropping strategies.
    
    Args:
        data_dir (str): Directory containing .npz files
        img_size (int): Target image size after resizing
        crop_size (int, optional): Size for cropping before resize. If None, no cropping.
        crop_type (str): Type of cropping - 'center', 'random', 'corner', 'multi_crop'
        crop_position (str, optional): Specific position for corner cropping - 
                                     'top_left', 'top_right', 'bottom_left', 'bottom_right'
        image_key (str): Key name for images in .npz files
        label_key (str, optional): Key name for labels in .npz files
        file_format (str): File format to search for
        augment (bool): Enable data augmentation during cropping
        augment_type (str, optional): Type of augmentation to apply (e.g., 'fliplr')
    
    Crop Types:
        - 'center': Center crop (default)
        - 'random': Random crop (different each time)
        - 'corner': Crop from specified corner
        - 'multi_crop': Generate multiple crops from same image
        - 'smart': Content-aware cropping (experimental)
    
    Returns:
        tf.data.Dataset: Preprocessed TensorFlow dataset
    """
    self.CLIP_MAX = 1.0
    self.CLIP_MIN = -1.0
    self.img_size = img_size
    self.crop_size = crop_size
    self.crop_type = crop_type.lower()
    self.crop_position = crop_position
    self.augment = augment
    self.augment_type = augment_type
    self.data_dir = os.path.abspath(data_dir)
    self.image_key = image_key
    self.label_key = label_key
    
    # Validate crop parameters
    valid_crop_types = ['center', 'random', 'corner', 'multi_crop', 'smart']
    assert self.crop_type in valid_crop_types, f"crop_type must be one of {valid_crop_types}"
    
    if self.crop_type == 'corner':
        valid_positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        assert self.crop_position in valid_positions, f"crop_position must be one of {valid_positions} for corner cropping"
    
    if self.crop_type == 'multi_crop' and self.crop_size is None:
        raise ValueError("crop_size must be specified for multi_crop mode")
    assert os.path.exists(self.data_dir), f"data_dir {self.data_dir} not exists"
    assert os.path.isdir(self.data_dir), f"data_dir {self.data_dir} is not a directory"
    assert self.image_key is not None, "image_key should not be None"
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

  def _apply_crop(self, arr, is_training=True):
    """
    Apply various cropping strategies to the input array.
    
    Args:
        arr (np.ndarray): Input image array [H, W, C]
        is_training (bool): Whether in training mode (affects random cropping)
    
    Returns:
        np.ndarray: Cropped image array
    """
    if self.crop_size is None:
      return arr
    
    h, w, c = arr.shape
    crop_size = self.crop_size
    
    # Ensure crop size is valid
    if crop_size > min(h, w):
      # If crop size is larger than image, pad the image
      pad_h = max(0, crop_size - h)
      pad_w = max(0, crop_size - w)
      arr = np.pad(arr, ((pad_h//2, pad_h - pad_h//2), 
                        (pad_w//2, pad_w - pad_w//2), 
                        (0, 0)), mode='reflect')
      h, w, c = arr.shape
    
    if self.crop_type == 'center':
      return self._center_crop(arr, crop_size)
      
    elif self.crop_type == 'random' and is_training:
      return self._random_crop(arr, crop_size)
      
    elif self.crop_type == 'corner':
      return self._corner_crop(arr, crop_size, self.crop_position)
      
    elif self.crop_type == 'multi_crop':
      # For multi_crop, return center crop (could be extended to return multiple crops)
      return self._center_crop(arr, crop_size)
      
    elif self.crop_type == 'smart':
      return self._smart_crop(arr, crop_size)
      
    else:
      # Default to center crop
      return self._center_crop(arr, crop_size)

  def _center_crop(self, arr, crop_size):
    """Center crop the image."""
    h, w, _ = arr.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return arr[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

  def _random_crop(self, arr, crop_size):
    """Random crop the image."""
    h, w, _ = arr.shape
    start_h = np.random.randint(0, h - crop_size + 1)
    start_w = np.random.randint(0, w - crop_size + 1)
    return arr[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

  def _corner_crop(self, arr, crop_size, position):
    """Crop from a specific corner."""
    h, w, _ = arr.shape
    
    if position == 'top_left':
      start_h, start_w = 0, 0
    elif position == 'top_right':
      start_h, start_w = 0, w - crop_size
    elif position == 'bottom_left':
      start_h, start_w = h - crop_size, 0
    elif position == 'bottom_right':
      start_h, start_w = h - crop_size, w - crop_size
    else:
      # Default to center if position is invalid
      start_h = (h - crop_size) // 2
      start_w = (w - crop_size) // 2
      
    return arr[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

  def _smart_crop(self, arr, crop_size):
    """
    Smart crop that tries to find the most interesting region.
    Uses simple variance-based content detection.
    """
    h, w, _ = arr.shape
    
    # If image is grayscale, convert to single channel for processing
    if arr.shape[-1] == 1:
      gray = arr[:, :, 0]
    else:
      # Convert to grayscale using standard weights
      gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    
    best_variance = -1
    best_start_h, best_start_w = 0, 0
    
    # Sample multiple positions and choose the one with highest variance (most content)
    num_samples = min(25, (h - crop_size + 1) * (w - crop_size + 1))
    
    for _ in range(num_samples):
      start_h = np.random.randint(0, h - crop_size + 1)
      start_w = np.random.randint(0, w - crop_size + 1)
      
      crop_region = gray[start_h:start_h + crop_size, start_w:start_w + crop_size]
      variance = np.var(crop_region)
      
      if variance > best_variance:
        best_variance = variance
        best_start_h, best_start_w = start_h, start_w
    
    return arr[best_start_h:best_start_h + crop_size, 
              best_start_w:best_start_w + crop_size, :]

  def _apply_augmentation(self, arr, augment_type='fliplr'):
    """
    Apply data augmentation if enabled.
    
    Args:
        arr (np.ndarray): Input image array
    
    Returns:
        np.ndarray: Augmented image array
    """
    if not self.augment:
      return arr
    else:
      if self.augment_type == 'fliplr':
        # Random horizontal flip
        if np.random.rand() > 0.5:
          arr = np.fliplr(arr)
      elif self.augment_type == 'flipud':
        # Random vertical flip
        if np.random.rand() > 0.5:
          arr = np.flipud(arr)
      elif self.augment_type == 'rotate90':
        if np.random.rand() > 0.5:
          # Rotate 90 degrees clockwise
          arr = np.rot90(arr)  # Rotate 90 degrees clockwise
      else:
        raise ValueError(f"Unknown augment_type: {self.augment_type}")
    
    # Additional augmentations can be added here
    # For example:
    # Random brightness adjustment
    #if np.random.rand() > 0.5:
    #  brightness_factor = np.random.uniform(0.8, 1.2)
    #  arr = np.clip(arr * brightness_factor, self.CLIP_MIN, self.CLIP_MAX) 
    # Random small rotation (±5 degrees)
    #if np.random.rand() > 0.7:
    #  angle = np.random.uniform(-5, 5)
      # Simple rotation using numpy (for small angles)
      # For more complex augmentations, could use tf.image or other libraries
      
    return arr

  def _load_npz(self, path, is_training=True):

    def _preprocess(x):
      raw_name = x.decode('utf-8')
      data = np.load(raw_name)
      assert self.image_key in list(data.keys())
      arr = data[self.image_key].astype(np.float32)
      
      # Handle different image dimensions
      if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=-1)
      elif len(arr.shape) == 4:
        # If batch dimension exists, take first image
        arr = arr[0]
      
      self.h, self.w, self.c = arr.shape
      
      # Load labels if available
      label = None
      if self.label_key is not None and self.label_key in data:
        label = data[self.label_key].astype(np.int32)
      
      # Apply cropping with the selected strategy
      if self.crop_size is not None:
        arr = self._apply_crop(arr, is_training=is_training)
      
      # Apply augmentation if enabled and in training mode
      if is_training and self.augment:
        arr = self._apply_augmentation(arr, self.augment_type)
      
      # Normalize to target range
      # Assume input is in [0, 1] range and convert to [CLIP_MIN, CLIP_MAX]
      arr = arr * (self.CLIP_MAX - self.CLIP_MIN) + self.CLIP_MIN
      
      if self.label_key is None:
        return arr
      else:
        return arr, label

    output_types = (tf.float32, tf.int32) if self.label_key is not None else tf.float32
    if self.label_key is not None:
      img, label = tf.numpy_function(_preprocess, [path], output_types)
      label = tf.ensure_shape(label, [])
    else:
      img = tf.numpy_function(_preprocess, [path], output_types)
    
    img_size = self.img_size if self.crop_size is None else self.crop_size
    img = tf.ensure_shape(img, [img_size, img_size, None])
    if self.label_key is not None:
      return img, label
    else:
      return img
    
  def _get_dataset(self):
    """
    Create training and validation datasets with appropriate preprocessing.
    
    Returns:
        tuple: (train_dataset, validation_dataset)
    """
    # Training dataset with augmentation and random cropping
    train_ds = (
      self.train_ds
      .map(lambda x: self._load_npz(x, is_training=True), 
           num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
      .shuffle(buffer_size=10000)
      .repeat()
    )
    
    # Validation dataset with deterministic preprocessing
    valid_ds = (
      self.valid_ds
      .map(lambda x: self._load_npz(x, is_training=False), 
           num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
    )
    
    return (train_ds, valid_ds)

  def get_multi_crop_dataset(self, num_crops=5):
    """
    Create a dataset with multiple crops per image for test-time augmentation.
    
    Args:
        num_crops (int): Number of crops to generate per image
    
    Returns:
        tf.data.Dataset: Dataset with multiple crops per image
    """
    def multi_crop_fn(path):
      crops = []
      for _ in range(num_crops):
        crop = self._load_npz(path, is_training=True)  # Use training=True for random crops
        crops.append(crop)
      return tf.stack(crops)
    
    return (
      self.valid_ds
      .map(multi_crop_fn, num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
    )

  def get_info(self):
    """
    Get information about the dataset.
    
    Returns:
        dict: Dataset information
    """
    return {
      'num_train_files': len(self.train_npzfiles),
      'num_valid_files': len(self.valid_npzfiles),
      'total_files': len(self.all_npzfiles),
      'data_dir': self.data_dir,
      'image_key': self.image_key,
      'label_key': self.label_key,
      'crop_type': self.crop_type,
      'crop_size': self.crop_size,
      'crop_position': self.crop_position,
      'augment': self.augment,
      'target_size': self.img_size,
    }


def unit_test():
  """
  Comprehensive unit test for the enhanced DataLoader with multiple cropping options.
  """
  print("=== DataLoader Unit Test ===")
  
  # Create sample data for testing
  os.makedirs("test_data", exist_ok=True)
  
  # Generate sample images with different sizes
  sample_images = []
  sample_labels = []
  
  for i in range(5):
    # Create images of different sizes to test cropping
    if i % 2 == 0:
      img = np.random.rand(128, 128, 3).astype(np.float32)  # RGB
    else:
      img = np.random.rand(96, 96, 1).astype(np.float32)   # Grayscale
    
    label = i % 3  # 3 classes
    
    sample_images.append(img)
    sample_labels.append(label)
    
    # Save as npz
    np.savez(f"test_data/sample_{i}.npz", 
             images=img, labels=label)
  
  print(f"Created {len(sample_images)} test samples")
  
  # Test different cropping strategies
  crop_configs = [
    {"crop_type": "center", "crop_size": 64},
    {"crop_type": "random", "crop_size": 64},
    {"crop_type": "corner", "crop_size": 64, "crop_position": "top_left"},
    {"crop_type": "smart", "crop_size": 64},
    {"crop_type": "center", "crop_size": None},  # No cropping
  ]
  
  for i, config in enumerate(crop_configs):
    print(f"\n--- Test {i+1}: {config} ---")
    
    try:
      # Create DataLoader with current config
      loader = DataLoader(
        data_dir="test_data",
        img_size=64,
        crop_size=config.get("crop_size"),
        crop_type=config.get("crop_type", "center"),
        crop_position=config.get("crop_position"),
        image_key="images",
        label_key="labels",
        augment=True if config.get("crop_type") == "random" else False,
        augment_type=config.get("augment_type", None),
      )
      
      # Get dataset info
      info = loader.get_info()
      print(f"Dataset info: {info}")
      
      # Test dataset creation
      train_ds, valid_ds = loader._get_dataset()
      
      # Test one batch
      for batch in train_ds.take(1):
        if isinstance(batch, tuple):
          images, labels = batch
          print(f"Batch shape: images={images.shape}, labels={labels.shape}")
        else:
          images = batch
          print(f"Batch shape: images={images.shape}")
        
        print(f"Image value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
      
      print("✓ Success")
      
    except Exception as e:
      print(f"✗ Error: {e}")
  
  # Test multi-crop functionality
  print(f"\n--- Multi-crop Test ---")
  try:
    loader = DataLoader(
      data_dir="test_data",
      img_size=64,
      crop_size=64,
      crop_type="random",
      image_key="images",
      label_key="labels"
    )
    
    multi_crop_ds = loader.get_multi_crop_dataset(num_crops=3)
    
    for batch in multi_crop_ds.take(1):
      if isinstance(batch, tuple):
        crops, labels = batch
        print(f"Multi-crop shape: {crops.shape}")  # Should be [3, H, W, C]
      else:
        crops = batch
        print(f"Multi-crop shape: {crops.shape}")
    
    print("✓ Multi-crop success")
    
  except Exception as e:
    print(f"✗ Multi-crop error: {e}")
  
  # Test error handling
  print(f"\n--- Error Handling Test ---")
  
  # Test invalid crop type
  try:
    loader = DataLoader(
      data_dir="test_data",
      img_size=64,
      crop_type="invalid_type",
      image_key="images"
    )
    print("✗ Should have failed with invalid crop type")
  except AssertionError:
    print("✓ Correctly caught invalid crop type")
  
  # Test corner crop without position
  try:
    loader = DataLoader(
      data_dir="test_data",
      img_size=64,
      crop_type="corner",
      image_key="images"
    )
    print("✗ Should have failed with missing crop position")
  except AssertionError:
    print("✓ Correctly caught missing crop position")
  
  # Cleanup
  import shutil
  shutil.rmtree("test_data")
  print("\n=== Test completed ===")


def demo_cropping_strategies():
  """
  Demonstrate different cropping strategies with visualizations.
  """
  print("=== Cropping Strategies Demo ===")
  
  # This would require matplotlib for visualization
  try:
    import matplotlib.pyplot as plt
    
    # Create a sample image with distinct regions
    img = np.zeros((200, 200, 3), dtype=np.float32)
    
    # Add colored regions to make cropping effects visible
    img[50:100, 50:100, 0] = 1.0  # Red square
    img[100:150, 100:150, 1] = 1.0  # Green square
    img[25:75, 125:175, 2] = 1.0   # Blue square
    
    # Add some noise for smart crop to detect
    img += np.random.normal(0, 0.1, img.shape)
    img = np.clip(img, 0, 1)
    
    crop_size = 80
    
    # Create a temporary DataLoader to test cropping
    os.makedirs("demo_data", exist_ok=True)
    np.savez("demo_data/demo.npz", images=img)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Test different crops
    crop_types = ['center', 'corner', 'smart']
    crop_configs = [
      {'crop_type': 'center'},
      {'crop_type': 'corner', 'crop_position': 'top_left'},
      {'crop_type': 'smart'}
    ]
    
    for i, (crop_type, config) in enumerate(zip(crop_types, crop_configs)):
      loader = DataLoader(
        data_dir="demo_data",
        img_size=crop_size,
        crop_size=crop_size,
        **config,
        image_key="images"
      )
      
      # Apply cropping
      cropped = loader._apply_crop(img, is_training=False)
      
      axes[0, i+1].imshow(cropped)
      axes[0, i+1].set_title(f'{crop_type.title()} Crop')
      axes[0, i+1].axis('off')
    
    # Show random crops
    loader = DataLoader(
      data_dir="demo_data",
      img_size=crop_size,
      crop_size=crop_size,
      crop_type='random',
      image_key="images"
    )
    
    for i in range(2):
      cropped = loader._apply_crop(img, is_training=True)
      axes[1, i].imshow(cropped)
      axes[1, i].set_title(f'Random Crop {i+1}')
      axes[1, i].axis('off')
    
    # Show augmented crop
    loader.augment = True
    cropped_aug = loader._apply_crop(img, is_training=True)
    cropped_aug = loader._apply_augmentation(cropped_aug)
    axes[1, 2].imshow(cropped_aug)
    axes[1, 2].set_title('Augmented Crop')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('cropping_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Cleanup
    import shutil
    shutil.rmtree("demo_data")
    
    print("✓ Cropping demo completed. Check 'cropping_demo.png' for results.")
    
  except ImportError:
    print("Matplotlib not available. Skipping visualization demo.")
  except Exception as e:
    print(f"Demo error: {e}") 


if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description='Test DataLoader with enhanced cropping')
  parser.add_argument('--test', action='store_true', help='Run unit tests')
  parser.add_argument('--demo', action='store_true', help='Run cropping demo')
  parser.add_argument('--all', action='store_true', help='Run all tests')
  
  args = parser.parse_args()
  
  if args.test or args.all:
    unit_test()
  
  if args.demo or args.all:
    demo_cropping_strategies()
  
  if not any([args.test, args.demo, args.all]):
    print("Usage: python data_loader.py [--test] [--demo] [--all]")
    print("  --test: Run unit tests")
    print("  --demo: Run cropping strategies demo")  
    print("  --all:  Run all tests and demos")
