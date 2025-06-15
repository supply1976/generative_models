import torch
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import tqdm
import PIL
import numpy as np
import sys, os


def make_grid(pil_images, rows, cols):
  w, h = pil_images[0].size
  mode = pil_images[0].mode
  grid = PIL.Image.new(mode, size=(cols*w, rows*h))
  for i, image in enumerate(pil_images):
    grid.paste(image, box=(i%cols*w, i//cols*h))
  return grid


model_path = sys.argv[1]
#repo_id = "google/ddpm-cat-256"
model_id = os.path.join(model_path + "unet")
scheduler_id = os.path.join(model_path + "scheduler")

model = UNet2DModel.from_pretrained(model_id)
scheduler =  DDPMScheduler.from_pretrained(scheduler_id)

model.to("cuda:0")

# Process images in smaller batches
total_samples = 10
batch_size = 10
all_samples = []

for _ in range(total_samples // batch_size):
    samples = torch.randn(batch_size, 3, 256, 256).to("cuda:0")
    with torch.no_grad():
        for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
            residual = model(samples, t).sample
            samples = scheduler.step(residual, t, samples).prev_sample
    all_samples.append(samples.cpu())

# Concatenate all samples
final_samples = torch.cat(all_samples, dim=0)
print(final_samples.shape, final_samples.max(), final_samples.min())
images = final_samples.permute(0, 2, 3, 1).numpy()
images = np.clip(images, -1.0, 1.0)
images = 0.5 * (images + 1.0)

print(images.shape, images.max(), images.min())
np.savez_compressed(os.path.join(model_path, "dmgen_images.npz"), images=images)


# save to png
images = (images * 255.0).astype(np.uint8)
images = [PIL.Image.fromarray(x) for x in images]
image_grid = make_grid(images, rows=4, cols=4)
image_grid.show()


