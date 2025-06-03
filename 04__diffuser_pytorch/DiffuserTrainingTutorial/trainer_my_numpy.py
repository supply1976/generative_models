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


@dataclass(order=True, frozen=True)
class TrainingConfig:
  image_size: int = 512
  train_batch_size: int = 8
  eval_batch_size: int = 8
  num_epochs: int = 20
  gradient_accumulation_steps: int = 1
  learning_rate: float = 1.0e-4
  lr_warmup_steps: int = 500
  save_image_epochs: int = 10
  save_model_epochs: int = 30
  mixed_precision: str = 'no'
  output_dir: str = 'mydataset_metal_test1-512'
  overwrite_output_dir: bool = True
  seed: int = 0


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

  image_grid = make_grid(images, rows=4, cols=4)

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
    progress_bar.set_description(("Epoch").format(epoch))

    for step, batch in enumerate(train_dataloader):
      images = batch['images']
      noise = torch.randn(images.shape).to(images.device)
      bs = images.shape[0]

      timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=images.device).long()

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
        pipeline.save_pretrained(config.output_dir)


config = TrainingConfig()

#dataset_name = "huggan/smithsonian_butterflies_subset"
#dataset = load_dataset(dataset_name, split='train')
#dataset.set_transform(png_transform)

npz_file = "/home/tacowu/mydatasets/metal_test1_713/all_images_713x512x512x1.npz"
images = np.load(npz_file)['images']



#######
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
  sample_size=config.image_size,
  in_channels=3,
  out_channels=3,
  layers_per_block=2,
  block_out_channels=(128, 128, 256, 256, 512, 512),
  down_block_types=(
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "AttnDownBlock2D",
    "DownBlock2D",
  ),
  up_block_types=(
    "UpBlock2D",
    "AttnUpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
  ),
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)

noise_scheduler = DDPMScheduler()
noise_scheduler.config.num_train_timesteps: int=1000

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

#lr_scheduler = get_cosine_schedule_with_warmup(
#  optimizer=optimizer,
#  num_warmup_steps=config.lr_warmup_steps,
#  num_training_steps=(len(train_dataloader) * config.num_epochs)
#)

lr_scheduler = get_constant_schedule(
  optimizer=optimizer,
)

train_loop(config, noise_scheduler, model, optimizer, train_dataloader, lr_scheduler)
