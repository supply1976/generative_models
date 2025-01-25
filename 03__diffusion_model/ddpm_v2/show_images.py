import os, sys
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'

import matplotlib.pyplot as plt

data = np.load(sys.argv[1])
images = data['images']
print(images.shape, images.max(), images.min(), images.std(), images.mean())

if len(images.shape)==3: 
  images = np.expand_dims(images, axis=-1)
nums, h, w, c = images.shape

images = np.clip(images, -1, 1)
images = 0.5*(images+1)

nrows, ncols = (5,5)
num_figs = nums // (nrows*ncols) 
num_figs = num_figs if nums%(nrows*ncols)==0 else num_figs + 1
axess = []
for n in range(num_figs):
  if n>=5: break
  fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(8,8))
  axes = axes.flatten()
  axess.extend(axes)

for i, img in enumerate(images):
  axess[i].imshow(img)
  if i==len(axess)-1: break



plt.tight_layout()
plt.show()

