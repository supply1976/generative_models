import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import yaml

  
csv_files = sys.argv[1:]

yaml_fn = "training_config.yaml"
fig, ax = plt.subplots(figsize=(12,9))

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
  dfloss.plot(
    x='epoch', y='loss', 
    logy=True, ax=ax, grid=True, label=label, alpha=0.7)

ax.legend()
plt.show()
