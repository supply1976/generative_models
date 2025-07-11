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


def create_base_df(gen_csv_files):
  trcond_cols = ["net", 'scheduler', "loss", "pred", "steps", 'imgen_epoch',
    "clipDN", "gen_date", "batch", "LR", 'attentions', 'num_heads']
  
  df = pd.DataFrame(columns=trcond_cols)

  for i, csv in enumerate(gen_csv_files):
    pdir = os.path.dirname(csv)
    ppdir = os.path.dirname(pdir)
    print(pdir)
    rev_steps = (os.path.basename(pdir)).split("_")[1]
    gen_date = (os.path.basename(pdir)).split("_")[-1]

    yaml_fn = os.path.join(ppdir, "training_config.yaml")
    
    with open(os.path.join(pdir, "imgen.log"), 'r') as f_imlog:
      imgen_log = f_imlog.readlines()
    imgen_log = [_.strip() for _ in imgen_log]
    imgen_model = os.path.basename(imgen_log[0].split(":")[-1])
    imgen_model = imgen_model.split(".")[-2]
    imgen_epoch = imgen_model.split("_")[-1]

    with open(yaml_fn, 'r') as f:
      yaml_dict = yaml.safe_load(f)
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
      rev_steps, imgen_epoch, clipDN, gen_date, batch, lr, atts, nhs]
    df.loc[i] = tr_cond
  return df


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--real_csv', type=str, default=None)
  parser.add_argument('--workdir', type=str, default=None)
  FLAGS, _ = parser.parse_known_args()
  assert FLAGS.real_csv is not None
  assert FLAGS.workdir is not None
  
  df_real = pd.read_csv(FLAGS.real_csv, sep='\s+')
  
  cols_gradImgSE = ['Hx', 'Hy']
  cols_psdSE = ['S0x', 'S0y']
  cols_gzip = ['gzipSize', 'Pden']
  cols_LZCxy = ['LZC_x', 'LZC_y']
  cols_inceptV3 = [_ for _ in list(df_real) if _.startswith('inceptV3')]

  feats_gradImgSE_real = df_real[cols_gradImgSE].values
  feats_psdSE_real     = df_real[cols_psdSE].values
  feats_gzip_real      = df_real[cols_gzip].values
  feats_LZCxy_real     = df_real[cols_LZCxy].values
  feats_inceptV3_real  = df_real[cols_inceptV3].values

  gen_csv_files = []
  for root, dirs, files in os.walk(FLAGS.workdir):
    for fn in files:
      if fn.endswith("_features.csv"):
        csv_file = os.path.join(root, fn)
        gen_csv_files.append(csv_file)
  
  print("#"*80)
  try:
    df = create_base_df(gen_csv_files)
  except:
    df = pd.DataFrame()
  
  values_FID=[]
  values_KLD_gradImgSE=[]
  values_KLD_psdSE=[]
  values_KLD_LZC=[]
  values_KLD_gzip=[]
  for i, gen_csv in enumerate(gen_csv_files):
    print(i, "/", len(gen_csv_files))
    df_gen = pd.read_csv(gen_csv, sep='\s+')
    feats_gradImgSE_gen = df_gen[cols_gradImgSE].values
    feats_psdSE_gen     = df_gen[cols_psdSE].values
    feats_gzip_gen      = df_gen[cols_gzip].values
    feats_LZCxy_gen     = df_gen[cols_LZCxy].values
    feats_inceptV3_gen  = df_gen[cols_inceptV3].values
    fid_i = calc_FID(feats_inceptV3_real, feats_inceptV3_gen)
    kld_gradImgSE = _calc_KLD_use_gaussian_kde(feats_gradImgSE_real, feats_gradImgSE_gen)
    kld_psdSE = _calc_KLD_use_gaussian_kde(feats_psdSE_real, feats_psdSE_gen)
    kld_LZC = _calc_KLD_use_gaussian_kde(feats_LZCxy_real, feats_LZCxy_gen)
    kld_gzip = _calc_KLD_use_gaussian_kde(feats_gzip_real, feats_gzip_gen)

    values_FID.append(fid_i)
    values_KLD_gradImgSE.append(kld_gradImgSE)
    values_KLD_psdSE.append(kld_psdSE)
    values_KLD_LZC.append(kld_LZC)
    values_KLD_gzip.append(kld_gzip)

  df['KLD_gradImg'] = np.around(values_KLD_gradImgSE, 4)
  df['KLD_LZC'] = np.around(values_KLD_LZC, 4)
  df['KLD_fftPSD'] = np.around(values_KLD_psdSE, 4)
  df['KLD_GzipSize'] = np.around(values_KLD_gzip, 4)
  df['FID'] = np.around(values_FID, 4)
  print(df)
  df.to_csv(os.path.join(FLAGS.workdir, "eval_results.csv"), sep="\t", index=False)
  for csv in gen_csv_files:
    print(csv)


def _calc_KLD_use_gaussian_kde(data_real, data_gen, show_dist=False):
  #print(data_real.shape)
  #print(data_gen.shape)
  xmin, ymin = np.min(np.vstack([data_real, data_gen]), axis=0)
  xmax, ymax = np.max(np.vstack([data_real, data_gen]), axis=0)
  X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
  grid_coords = np.vstack([X.flatten(), Y.flatten()])
  kde1 = sps.gaussian_kde(data_real.T)
  kde2 = sps.gaussian_kde(data_gen.T)
  pdf1 = kde1(grid_coords)
  pdf2 = kde2(grid_coords)
  pdf1 = pdf1 / np.sum(pdf1)
  pdf2 = pdf2 / np.sum(pdf2)
  
  # KLD(P||Q)
  #KLD_12 = sps.entropy(pk=pdf1, qk=pdf2)
  KLD_12 = tf.keras.losses.kld(pdf1, pdf2)
  KLD_12 = KLD_12.numpy()

  if show_dist:
    pdf1 = np.reshape(pdf1.T, X.shape)
    pdf2 = np.reshape(pdf2.T, X.shape)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4), 
            #sharex=True, sharey=True,
            )
    axes[0].plot(data_real[:,0], data_real[:,1], 'b.', markersize=2)
    axes[0].plot(data_gen[:,0], data_gen[:,1], 'r.', markersize=2)
    axes[0].set_title("KLD(P||Q)={}".format(np.around(KLD_12, 6)))
    axes[0].grid()
    axes[0].set_aspect('auto')
    # P
    axes[1].imshow(np.rot90(pdf1), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax],
          )
    axes[1].set_aspect('auto')
    axes[1].set_title("P")
    axes[1].plot(data_real[:,0], data_real[:,1], 'b.', markersize=2)
    # Q
    axes[2].imshow(np.rot90(pdf2), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax],
          )
    axes[2].set_title("Q")
    axes[2].plot(data_gen[:,0], data_gen[:,1], 'r.', markersize=2)
    axes[2].set_aspect('auto')
  return KLD_12


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
  plt.show()


