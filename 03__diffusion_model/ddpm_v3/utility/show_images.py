import os, sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
import argparse
#from skimage.transform import resize


def _get_img_grad_hist(images, bins):
  images = images[:,:,:,0]
  images = (images*15).astype(np.uint8)
  fy, fx = np.gradient(images, axis=(1,2))
  assert len(fx.shape)==3
  assert len(fy.shape)==3
  # remove edge
  fx = fx[:, 1:-1, 1:-1]
  fy = fy[:, 1:-1, 1:-1]
  # histogram counts
  pdf_fx = np.array([np.histogram(_, bins=bins)[0] for _ in fx])
  pdf_fy = np.array([np.histogram(_, bins=bins)[0] for _ in fy])
  return (pdf_fx, pdf_fy)


def display_images(images, nrows, ncols):
  if len(images.shape)==3: 
    images = np.expand_dims(images, axis=-1)
  nums, h, w, c = images.shape

  if c==2:
    images = np.concatenate([images, np.zeros([nums, h, w, 1])], axis=-1)

  num_figs = nums // (nrows*ncols) 
  num_figs = num_figs if nums%(nrows*ncols)==0 else num_figs + 1

  figs, img_axess, gradx_axess, grady_axess = ([], [], [], [])
  for n in range(num_figs):
    if n>=4: break
    fig1, img_axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,figsize=(8,6))
    #fig2, gradx_axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(14,8))
    #fig3, grady_axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(14,8))
    if nrows*ncols > 1:
      img_axes = img_axes.flatten()
    else:
      img_axes = [img_axes]
    #gradx_axes = gradx_axes.flatten()
    #grady_axes = grady_axes.flatten()
    img_axess.extend(img_axes)
    #gradx_axess.extend(gradx_axes)
    #grady_axess.extend(grady_axes)
    plt.tight_layout()

  for i, img in enumerate(images):
    img_axess[i].imshow(img, cmap='gray')
    if c==2 or c==3:
      img_axess[i].contour(img[:,:,0], levels=[0.5], colors='red')
      img_axess[i].contour(img[:,:,1], levels=[0.5], colors='blue')
    if c==3:
      img_axess[i].contour(img[:,:,2], levels=[0.5], colors='green')
    #img_axess[i].set_axis_off()

    #gradx_axess[i].plot(pdf_fx[i], '-o')
    
    #if FLAGS.feats_csv is not None:
    #  img_axess[i].set_title(str(feats[i]))
    
    #axess[i].set_title(r'$\theta$={}$\pi$'.format(np.around(i*0.05, 2))) 
    #axess[i].set_title("seed={}".format(i))
    if i==len(img_axess)-1: break


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ij', nargs=2, default=[1, 1], type=int)
  parser.add_argument('npz', type=str)
  parser.add_argument('--feats_csv', type=str, default=None)
  FLAGS, _ = parser.parse_known_args()
  
  data = np.load(FLAGS.npz)
  print(list(data.keys()))
  images = data['images']
  print(images.shape, images.max(), images.min(), images.std(), images.mean())

  if FLAGS.feats_csv is not None:
    df = pd.read_csv(FLAGS.feats_csv, sep='\s+')
    df['Hx'] = np.around(df.Hx.values, 2)
    df['Hy'] = np.around(df.Hy.values, 2)
    df = df.sort_values(['Hx', 'Hy'])
    images = images[df.index.values]
    feats = np.around(df[['Hx','Hy']].values, 2)
    print(feats.shape)

  i, j = FLAGS.ij

  #display_images(images_resized, 15, 20)
  display_images(images, i, j)
  
  #images_t = data['images_t_300']
  #display_images(images_t, 4, 5)


main()
plt.show()

