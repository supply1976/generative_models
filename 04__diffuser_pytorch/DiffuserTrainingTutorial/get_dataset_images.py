import os, sys
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import transforms


image_size = 128


def png_transform(batch):
  preprocess = transforms.Compose(
    [
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ]
  )
  images = [preprocess(image.convert("RGB")) for image in batch['image']]
  return {"images": images}


dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(dataset_name, split='train')

images = []
for img in dataset['image']:
  # img is PIL Image
  img = img.convert("RGB")
  
  # transforms.Resize return PIL Image (0 ~ 255, uint8)
  img = transforms.Resize((image_size, image_size))(img)
  
  # ToTensor() return torch tensor (channel first), normalize to (0 ~ 1), FP32
  #img = transforms.ToTensor()(img)
  #img = transforms.Normalize([0.5], [0.5])(img)
  
  # directly convert PIL to numpy 
  img = np.array(img).astype(np.float32)
  img =  img / 255.0
  
  images.append(img)
  #print(img.shape, img.dtype, img.max(), img.min())

images = np.stack(images, axis=0)
print(images.shape, images.dtype, images.max(), images.min())

np.savez_compressed(
  "butterflies_"+"x".join(map(str, images.shape)),
  images=images)

