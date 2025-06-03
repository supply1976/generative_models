import torch
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import tqdm
import PIL
import numpy as np

#repo_id = "google/ddpm-cat-256"
model_id = "./ddpm-butterflies-128/unet"
scheduler_id = "./ddpm-butterflies-128/scheduler"
model = UNet2DModel.from_pretrained(model_id)
scheduler =  DDPMScheduler.from_pretrained(scheduler_id)

#sample = torch.randn(16, 3, 256, 256)
sample = torch.randn(16, 3, 128, 128)


model.to("cuda:0")
sample = sample.to("cuda:0")
print(model.device)
print(sample.device)

def make_grid(pil_images, rows, cols):
  w, h = pil_images[0].size
  grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
  for i, image in enumerate(pil_images):
    grid.paste(image, box=(i%cols*w, i//cols*h))
  return grid



for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
  with torch.no_grad():
    residual = model(sample, t).sample
  sample = scheduler.step(residual, t, sample).prev_sample

print(sample.shape, sample.max(), sample.min())
images = sample.cpu().permute(0, 2, 3, 1)
images = (images + 1.0) * 127.5
images = images.numpy().astype(np.uint8)

images = [PIL.Image.fromarray(x) for x in images]
image_grid = make_grid(images, rows=4, cols=4)
image_grid.show()
