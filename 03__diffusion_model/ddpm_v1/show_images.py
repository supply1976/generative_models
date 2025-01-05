import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data = np.load(sys.argv[1])
images = data['images']
print(images.shape, images.max(), images.min(), images.std(), images.mean())

if len(images.shape)==3: 
  images = np.expand_dims(images, axis=-1)
nums, h, w, c = images.shape

#if c==2:
#  images = np.concatenate([images, np.zeros([nums, h, w, 1])], axis=-1)

ncols =9
#nrows = 8
nrows = 1 + nums//ncols if nums%ncols!=0 else nums//ncols
nrows = 5 if nrows > 5 else nrows 

for ci in range(c):
  fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(12,8))
  axes = axes.flatten()
  for i, img in enumerate(images):
    axes[i].imshow(img[:,:,ci], origin='lower', cmap='gray')
    #axes[i].axis('off')
    #axes[i].set_title(str(i))
    if i==len(axes)-1: break

for i, img in enumerate(images):
  fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8,4))
  axes[0].imshow(img[:,:,0], origin='lower', cmap='gray')
  axes[1].imshow(img[:,:,1], origin='lower', cmap='gray')
  if i>10: break

plt.tight_layout()
plt.show()

