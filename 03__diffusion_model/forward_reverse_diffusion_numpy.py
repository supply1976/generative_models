import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit


class GaussianDiffusion:
  def __init__(self, beta_ini=1e-4, beta_end=0.02, timesteps=1000):
    self.beta_ini = beta_ini
    self.beta_end = beta_end
    self.timesteps = timesteps
    self.beta = np.linspace(beta_ini, beta_end, timesteps, dtype=np.float64)
    self.alpha = 1.0 - self.beta
    self.alpha_bar = np.cumprod(self.alpha)
    self.alpha_bar_prev = np.append(1.0, self.alpha_bar[:-1])
    self.back_var = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
    self.back_mean_coef1 = np.sqrt(self.alpha_bar_prev) * self.beta / (1.0-self.alpha_bar)
    self.back_mean_coef2 = np.sqrt(self.alpha)*(1.0-self.alpha_bar_prev) / (1.0-self.alpha_bar)
    df = pd.DataFrame(columns=['t_index', 'beta', 'alpha_bar'])
    df['t_index'] = range(1, self.timesteps+1)
    df['beta'] = self.beta
    df['alpha_bar'] = self.alpha_bar
    df['back_mean_coef1'] = self.back_mean_coef1
    df['back_mean_coef2'] = self.back_mean_coef2
    df['bakc_var'] = self.back_var
    self.df = df
    print(df.head())
    print(df.tail())
    
  def forward_diffuse_sim(self, x0, t):
    t=t-1
    self.x0 = x0
    batch, h, w, c = x0.shape 
    noise_t = np.random.randn(batch, h, w, c)
    xt_mean = np.sqrt(self.alpha_bar[t]) * x0
    xt_sigma =  np.sqrt(1.0-self.alpha_bar[t])
    xt_sample = xt_mean + xt_sigma * noise_t
    assert xt_sample.shape == x0.shape
    back_mean_xtn1_xt_x0 = x0 * self.back_mean_coef1[t] + xt_sample * self.back_mean_coef2[t]
    return (xt_sample, back_mean_xtn1_xt_x0, noise_t)

  def reverse_diffuse_sim(self, xt, pred_x0, t):
    #pred_x0 = (xT - pred_noise * np.sqrt(1-self.alpha_bar[t])) / np.sqrt(self.alpha_bar[t])
    model_mean_t = pred_x0 * self.back_mean_coef1[t] + xt * self.back_mean_coef2[t]
    model_var_t = self.back_var[t]
    xt = model_mean_t + np.random.randn(*xt.shape) * np.sqrt(model_var_t)
    return xt

def myfunc(x):
  #val = a + b*x + c/(1+np.exp(-d*(x-e)))
  #val = a + b*x + c*sp.special.erf(d*(x-e))
  val = 4*np.tanh(x-0.3)
  return val

def rescale_clip_for_plot(images):
  images = images - images.min()
  images = images / images.max()
  print(images.max(), images.min())
  return images

#data = np.load("/home/taco/image_mad_sci_1000x1000.npy")
#data = [data, np.fliplr(data), np.flipud(data), np.rot90(data)]
#train_images = np.stack(data)
#train_images = data
#train_images = train_images[:, 0:250, 550:800, :]
#print(train_images.shape)

data = tf.keras.datasets.fashion_mnist.load_data()
data = tf.keras.datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_label) = data
# normalize images to (-1,1)
train_images = train_images.astype(np.float32)/127.5 - 1
train_images = np.clip(train_images, -1.0, 1.0)


gd = GaussianDiffusion(beta_end=0.02, timesteps=1000)

x0 = train_images[0:6]
x1, _, _ = gd.forward_diffuse_sim(x0, 1)
x10, _, _ = gd.forward_diffuse_sim(x0, 10)
x100, _, _ = gd.forward_diffuse_sim(x0, 100)


zT = np.random.randn(*x0.shape)
for t in reversed(range(0,1000)):
  x0_reverse = gd.reverse_diffuse_sim(zT, zT, t)
  zT = x0_reverse

#df.plot(x='t_index', y='rev_xt', style='-.', ax=ax2, grid=True)
#df.plot(x='t_index', y='model_mean_t', style='-', ax=ax3, grid=True)
#df.plot(x='t_index', y='true_mean', style='-', ax=ax3, grid=True)
#popt, pcov = curve_fit(myfunc, x, y, p0=[1, 1, 1, 10, 0.5])
#x = df_all.xt_sample.values
#y = df_all.true_noise.values
#z = np.polyfit(x, y, 1)
#print(z)
#fit_func = np.poly1d(z)
#fit_func = myfunc
#model_y = fit_func(x) 

#reverse_images = gd.reverse_path_sim(fit_func, np.random.randn(*train_images.shape))

x0 = rescale_clip_for_plot(x0)
x1 = rescale_clip_for_plot(x1)
x10 = rescale_clip_for_plot(x10)
x100 = rescale_clip_for_plot(x100)
x0_reverse = rescale_clip_for_plot(x0_reverse)

fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(10,6))
for i in range(6):
  axes[0, i].imshow(x0[i])
  axes[0, i].axis('off')
  axes[1, i].imshow(x1[i])
  axes[1, i].axis('off')
  axes[2, i].imshow(x10[i])
  axes[2, i].axis('off')
  axes[3, i].imshow(x100[i])
  axes[3, i].axis('off')
  axes[4, i].imshow(x0_reverse[i])
  axes[4, i].axis('off')

plt.tight_layout()
plt.show()

