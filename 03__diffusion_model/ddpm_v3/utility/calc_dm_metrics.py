import os, sys, argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.linalg import sqrtm
import yaml
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--real_csv', type=str, default=None)
  parser.add_argument('--workdir', type=str, default=None)
  FLAGS, _ = parser.parse_known_args()
  assert FLAGS.real_csv is not None
  assert FLAGS.workdir is not None
  gimgSE_real_feats, psdSE_real_feats, inceptV3_real_feats = read_features(FLAGS.real_csv)

  gen_csvs = []
  for root, dirs, files in os.walk(FLAGS.workdir):
    for fn in files:
      if fn.endswith("_features.csv"):
        csv_file = os.path.join(root, fn)
        gen_csvs.append(csv_file)
  
  sigma = 0.02
  print("#"*80)
  cols = ["KLD_gimg", "KLD_psd", "FID_inceptV3", 
    "net", 'scheduler', "loss", "pred", "steps", 
    "clipDN", "gen_date", "batch", "repeat", "LR",
    'attentions', 'num_heads',
    ]
  df = pd.DataFrame(columns=cols)

  for i, csv in enumerate(gen_csvs):
    pdir = os.path.dirname(csv)
    ppdir = os.path.dirname(pdir)
    print(pdir)
    rev_steps, gen_date = (os.path.basename(pdir)).split("_")[1:]
    yaml_fn = os.path.join(ppdir, "training_config.yaml")
    with open(yaml_fn, 'r') as f:
      yaml_dict = yaml.safe_load(f)
    dsrept = yaml_dict['DATASET']['REPEAT']
    pred_type = yaml_dict['TRAINING']['PRED_TYPE']
    loss_fn = yaml_dict['TRAINING']['LOSS_FN']
    #
    hp_dict = yaml_dict['TRAINING']['HYPER_PARAMETERS']
    batch = hp_dict['BATCH_SIZE']
    lr = hp_dict['LEARNING_RATE']
    # 
    net_dict = yaml_dict['TRAINING']['NETWORK']
    nrb = net_dict['NUM_RES_BLOCKS']
    sch = net_dict['SCHEDULER']
    gn = net_dict['NORM_GROUPS']
    c1 = net_dict['FIRST_CHANNEL']
    cm = net_dict['CHANNEL_MULTIPLIER']
    cm = "".join(map(str, cm))
    bk = net_dict['BLOCK_SIZE']
    #
    atts = "".join([str(int(x)) for x in net_dict['HAS_ATTENTION']])
    if 'NUM_HEADS' not in net_dict.keys():
      net_dict['NUM_HEADS']=1
    nhs = net_dict['NUM_HEADS']
    nn_code = ["unet", str(c1), "m", cm, "g", str(gn), "rb", str(nrb), "bk", str(bk)]
    nn_code = "".join(nn_code)
    if "_raw_" in csv:
      clipDN = 0
    else:
      clipDN = 1
    tr_cond = [
      nn_code, sch, loss_fn, pred_type, 
      rev_steps, clipDN, gen_date, batch, 
      dsrept, lr, atts, nhs]
    # get features
    gimgSE_gen_feats, psdSE_gen_feats, inceptV3_gen_feats = read_features(csv)
    # calc metrics
    kld_gimgSE = np.around(calc_KLD(gimgSE_real_feats, gimgSE_gen_feats, sigma=sigma), 2)
    kld_psdSE = np.around(calc_KLD(psdSE_real_feats, psdSE_gen_feats, sigma=sigma), 2)
    fid_inceptV3 = np.around(calc_FID(inceptV3_real_feats, inceptV3_gen_feats), 2)
    print(i, csv)
    L = [kld_gimgSE, kld_psdSE, fid_inceptV3]+tr_cond
    df.loc[i] = L
  print(df)
  df.to_csv(os.path.join(FLAGS.workdir, "eval_results.csv"), sep="\t", index=False)



def read_features(csv_file):
  df = pd.read_csv(csv_file, sep='\s+')
  cols = list(df)
  cols_inceptV3 = [_ for _ in cols if _.startswith("inceptV3")]
  gimgSE_feats = df[['Hx', 'Hy']].values
  psdSE_feats = df[['S0x', 'S0y']].values
  inceptV3_feats = df[cols_inceptV3].values
  return (gimgSE_feats, psdSE_feats, inceptV3_feats)
  

def nd_gaus(x, mu, sigma):
  """
    N-dim gaus()
  """
  d2 = np.sum((x-mu)**2, axis=-1)
  val = np.exp(-d2 / (2*sigma**2))
  return val


def calc_KLD(real_features, gen_features, sigma):
  n, fdim = real_features.shape
  llx_P, urx_P = (real_features[:,0].min(), real_features[:,0].max())
  lly_P, ury_P = (real_features[:,1].min(), real_features[:,1].max())
  llx_Q, urx_Q = (gen_features[:,0].min(), gen_features[:,0].max())
  lly_Q, ury_Q = (gen_features[:,1].min(), gen_features[:,1].max())
  llx = np.min([llx_P, llx_Q])
  lly = np.min([lly_P, lly_Q])
  urx = np.max([urx_P, urx_Q])
  ury = np.max([ury_P, ury_Q])
  Lx = np.linspace(llx, urx, 100)
  Ly = np.linspace(lly, ury, 100)
  gridX, gridY = np.meshgrid(Lx, Ly)
  samples = np.array(list(zip(gridX.flatten(), gridY.flatten())))
  P = nd_gaus(samples[:,None,:], real_features[None,:,:], sigma)
  P = np.mean(P, axis=-1)
  Q = nd_gaus(samples[:,None,:], gen_features[None,:,:], sigma)
  Q = np.mean(Q, axis=-1)
  #for i, xi in enumerate(samples):
  #  pi = np.mean([nd_gaus(xi, xj, sigma) for xj in real_features])
  #  qi = np.mean([nd_gaus(xi, xk, sigma) for xk in gen_features])
  P = P / np.sum(P)
  Q = Q / np.sum(Q)
  KLD = tf.keras.losses.kld(P, Q)
  return KLD.numpy()

def calc_FID(real_features, fake_features):
  n_real, fdim_real = real_features.shape
  n_fake, fdim_fake = fake_features.shape
  #print(real_features.shape, fake_features.shape)
  # Calculate mean and covariance of real and generated activations
  mu_real = np.mean(real_features, axis=0)
  mu_fake = np.mean(fake_features, axis=0)
  sigma_real = np.cov(real_features, rowvar=False)
  sigma_fake = np.cov(fake_features, rowvar=False)
  # Calculate FID score
  ssdiff = np.sum((mu_real - mu_fake)**2)
  covmean = sqrtm(sigma_real.dot(sigma_fake))
  if np.iscomplexobj(covmean):
    covmean = covmean.real
  fid = ssdiff + np.trace(sigma_real + sigma_fake - 2 * covmean)
  return fid

if __name__=="__main__":
  main()


