import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 12})
import yaml


def exp_moving_avg(data, ema=0.999):
  ema_data = np.zeros_like(data)
  ema_data[0] = data[0]
  for i in range(1, len(data)):
    ema_data[i] = ema * ema_data[i-1] + (1-ema) * data[i]
  return ema_data
  


csv_files = sys.argv[1:]

yaml_fn = "training_config.yaml"
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,6), sharey=True)

for csv in csv_files:
  if yaml_fn in os.listdir(os.path.dirname(csv)):
    with open(os.path.join(os.path.dirname(csv), yaml_fn), 'r') as f:
      yaml_dict = yaml.safe_load(f)
    lr = yaml_dict['TRAINING']['HYPER_PARAMETERS']['LEARNING_RATE']
    bs = yaml_dict['TRAINING']['HYPER_PARAMETERS']['BATCH_SIZE']
    gn = yaml_dict['TRAINING']['NETWORK']['NORM_GROUPS']
    c1 = yaml_dict['TRAINING']['NETWORK']['FIRST_CHANNEL']
    cm = yaml_dict['TRAINING']['NETWORK']['CHANNEL_MULTIPLIER']
    cm = "".join(map(str, cm))
    nn = "".join(["c", str(c1), "m", cm, "gn" ,str(gn)])
    label = "LR={}, BS={}, unet{}".format(lr, bs, nn)
  else:
    label = None 
  
  dfloss = pd.read_csv(csv, sep=",")
  dfloss['ema999_loss'] = exp_moving_avg(dfloss.loss.values, ema=0.999)
  dfloss['ema99_loss'] = exp_moving_avg(dfloss.loss.values, ema=0.99)

  dfloss.plot(
    x='epoch', y='loss', logy=True, ax=axes[0], grid=True, alpha=0.7)
  dfloss.plot(
    x='epoch', y='ema999_loss', logy=True, ax=axes[1], grid=True, alpha=0.7)
  dfloss.plot(
    x='epoch', y='ema99_loss', 
    label=label, logy=True, ax=axes[2], grid=True, alpha=0.7)


axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].legend()

axes[1].set_title("ema=0.999")
axes[2].set_title("ema=0.99")
plt.show()
