from dataclasses import dataclass, field, asdict
from datasets import load_dataset
#from huggingface_hub.utils.tqdm import progress_bar_states
#from setuptools.build_meta import no_install_setup_requires
#from tensorflow.python.keras.distribute.distributed_training_utils_v1 import process_batch_and_step_size
from torchvision import transforms
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.optimization import get_constant_schedule
from accelerate import Accelerator
from tqdm.auto import tqdm
import os, sys, math
import PIL
import numpy as np


def png_transform(batch):
  preprocess = transforms.Compose(
    [
      transforms.Resize((config.image_size, config.image_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ]
  )
  #images = [preprocess(image.convert("RGB")) for image in batch['image']]
  images = [preprocess(image.convert("L")) for image in batch['image']]
  return {"images": images}


def make_grid(images, rows, cols):
  w, h = images[0].size
  grid = PIL.Image.new('L', size=(cols*w, rows*h))
  for i, image in enumerate(images):
    grid.paste(image, box=(i%cols*w, i//cols*h))
  return grid


def image_gen(config, epoch, pipeline):
  images = pipeline(
    batch_size = config.eval_batch_size,
    generator = torch.manual_seed(config.seed)
  )[0]

  image_grid = make_grid(images, rows=2, cols=2)

  test_dir = os.path.join(config.output_dir, "samples")
  os.makedirs(test_dir, exist_ok=True)
  image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, noise_scheduler, model, optimizer, train_dataloader, lr_scheduler):
  accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with='tensorboard',
    project_dir=os.path.join(config.output_dir, 'logs'),
  )
  model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler,
  )

  global_step = 0
  for epoch in range(config.num_epochs):
    progress_bar =  tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(("Epoch {}").format(epoch))

    for step, batch in enumerate(train_dataloader):
      images = batch['images']
      noise = torch.randn(images.shape).to(images.device)
      bs = images.shape[0]

      timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bs,), device=images.device).long()

      noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
      with accelerator.accumulate(model):
        pred_noise = model(noisy_images, timesteps)['sample']
        loss = F.mse_loss(pred_noise, noise)
        accelerator.backward(loss)

        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

      progress_bar.update(1)
      logs = {
        "loss": loss.detach().item(),
        "lr": lr_scheduler.get_last_lr()[0],
        "step": global_step,
      }
      progress_bar.set_postfix(**logs)
      accelerator.log(logs, step=global_step)
      global_step +=1

    if accelerator.is_main_process:
      pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

      if (epoch+1) % config.save_image_epochs == 0 or epoch == config.num_epochs -1:
        image_gen(config, epoch, pipeline)

      if (epoch+1) % config.save_model_epochs == 0 or epoch == config.num_epochs -1:
        pipeline.save_pretrained(config.output_dir)


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, X_np, y_np=None, transform=None):
    """
    Args:
       X_np (np.ndarray): Feature array of shape (N, …).
       y_np (np.ndarray): Label array of shape (N,).
       transform (callable, optional): Optional transform to apply to X.
    """
    self.X_np = X_np
    self.y_np = y_np
    self.transform = transform

    self.X = torch.from_numpy(X_np)
    if y_np is not None:
      assert len(X_np) == len(y_np), "Features and labels must have the same length"
      self.y = torch.from_numpy(y_np)  # dtype int64 if y_np is int64

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    # If you converted up front (Option A):
    x = self.X[idx]  # torch.FloatTensor
    if self.transform:
      # If you need to do any additional processing on x
      x = self.transform(x)

    if self.y_np is not None:
      label = self.y[idx]  # torch.LongTensor or FloatTensor
      return {'images': x, 'labels': label}
    else:
      return {'images': x}


class NPZdataset(torch.utils.data.Dataset):
  def __init__(self, folder_path):
    self.folder_path = folder_path
    self.files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    self.files.sort()  # Sort files to ensure consistent order

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    file_path = os.path.join(self.folder_path, self.files[idx])
    data = np.load(file_path)
    img = data['image']  # Assuming 'images' is the key in the npz file
    if img.ndim == 2:  # If the image is in HW format
      img = np.expand_dims(img, axis=-1)
    img = 2 * img - 1  # Normalize to [-1, 1]
    #print(img.shape, img.min(), img.max())

    img = np.transpose(img, (2, 0, 1))  # Convert to CHW format
    img_tensor = torch.tensor(img, dtype=torch.float32)
    
    return {'images': img_tensor}

    
@dataclass(order=True, frozen=True)
class TrainingConfig:
  image_size: int = 512
  image_channels: int = 2
  first_conv_channel: int = 128
  channel_multiplier: tuple = (1, 1, 2, 2, 4, 4)
  block_out_channels: tuple = (128, 128, 256, 256, 512, 512)
  train_batch_size: int = 4
  eval_batch_size: int = 4
  num_epochs: int = 500
  gradient_accumulation_steps: int = 1
  learning_rate: float = 1.0e-4
  lr_warmup_steps: int = 10000
  save_image_epochs: int = 50
  save_model_epochs: int = 50
  mixed_precision: str = 'no'
  output_dir: str = 'mydataset_IMEC_metalHV_512x2_run01'
  overwrite_output_dir: bool = True
  seed: int = 0


def main():
  training_config = TrainingConfig()
  #dataset_name = "huggan/smithsonian_butterflies_subset"
  #dataset = load_dataset(dataset_name, split='train')
  #dataset.set_transform(png_transform)
  
  #npz_file = "/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/metal_test1/pitch_8_512x512x1/all_images_713x512x512x1.npz"
  data_folder = "/remote/ltg_proj02_us01/user/richwu/datasets_for_ML_prototypes/IMEC_metalHV_for_lowNA_euv/npz_files/rast_ETCH_KERNEL"
  dataset = NPZdataset(data_folder)


  #######
  train_dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=training_config.train_batch_size, 
    shuffle=True)

  model = UNet2DModel(
    sample_size=training_config.image_size,
    in_channels=training_config.image_channels,
    out_channels=training_config.image_channels,
    flip_sin_to_cos = False,
    layers_per_block=2,
    block_out_channels=training_config.block_out_channels,
    down_block_types=(
      "DownBlock2D",
      "DownBlock2D",
      "DownBlock2D",
      "DownBlock2D",
      "DownBlock2D",
      #"AttnDownBlock2D",
      "DownBlock2D",
    ),
    up_block_types=(
      "UpBlock2D",
      "UpBlock2D",
      #"AttnUpBlock2D",
      "UpBlock2D",
      "UpBlock2D",
      "UpBlock2D",
      "UpBlock2D",
    ),
  )

  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print("total parameters:", total_params)

  noise_scheduler = DDPMScheduler()

  optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=training_config.learning_rate)

  lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=training_config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * training_config.num_epochs)
  )

  #lr_scheduler = get_constant_schedule(
  #  optimizer=optimizer,
  #)

  train_loop(
    training_config, 
    noise_scheduler, 
    model, 
    optimizer, 
    train_dataloader, 
    lr_scheduler)


if __name__ == "__main__":
  main()

# This code is designed to train a diffusion model using a custom dataset.
