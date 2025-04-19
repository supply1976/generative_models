import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt

"""
def calc_gaus_KLD(real_features, gen_features):
  # real_features as P
  # gen_features as Q
  n, fdim = real_features.shape
  mu_P = np.mean(real_features, axis=0)
  mu_Q = np.mean(gen_features, axis=0)
  mu_diff = mu_Q - mu_P
  cov_P = np.cov(real_features, rowvar=False)
  cov_Q = np.cov(gen_features, rowvar=False)
  invcov_P = np.linalg.inv(cov_P)
  invcov_Q = np.linalg.inv(cov_Q)
  det_P = np.linalg.det(cov_P)
  det_Q = np.linalg.det(cov_Q)
  val1 = np.trace(invcov_Q.dot(cov_P)) - fdim
  val2 = mu_diff.dot(invcov_Q.dot(mu_diff)) + np.log(det_Q/det_P)
  kld = val1 + val2
  return kld 
"""

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--real_csv', type=str, default=None)
  parser.add_argument('--workdir', type=str, default=None)
  FLAGS, _ = parser.parse_known_args()
  assert FLAGS.real_csv is not None
  assert FLAGS.workdir is not None
  
  df_real = pd.read_csv(FLAGS.real_csv, sep='\s+')

  gen_csvs = []
  for root, dirs, files in os.walk(FLAGS.workdir):
    for fn in files:
      if fn.endswith("_features.csv"):
        csv_file = os.path.join(root, fn)
        gen_csvs.append(csv_file)
  
  df_evals = pd.read_csv(os.path.join(FLAGS.workdir, "eval_results.csv"), sep='\s+')
  print(df_evals)
  qor = df_evals[['KLD_gimg', 'KLD_psd', 'FID_inceptV3']].values
  qor = np.around(qor, 2)
  
  print("#"*80)
  for i, csv in enumerate(gen_csvs):
    df_gen = pd.read_csv(csv, sep='\s+')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    axes = axes.flatten()
    df_real.plot(kind='scatter', x='Hx', y='Hy', s=4, ax=axes[0], label="trainset")
    df_real.plot(kind='scatter', x='Hx', y='Hy', s=4, ax=axes[1], label="trainset")
    df_gen.plot(kind='scatter', x='Hx', y='Hy', s=4, ax=axes[1], 
      color='red', alpha=0.5, label="gen")
    axes[1].set_title("KLD={}".format(qor[i][0]))

    df_real.plot(kind='scatter', x='S0x', y='S0y', s=4, ax=axes[2], label="trainset")
    df_real.plot(kind='scatter', x='S0x', y='S0y', s=4, ax=axes[3], label="trainset")
    df_gen.plot(kind='scatter', x='S0x', y='S0y', s=4, ax=axes[3], 
      color='red', alpha=0.5, label="gen")
    axes[3].set_title("KLD={}".format(qor[i][1]))
    plt.tight_layout()

if __name__=="__main__":
  main()
  #plt.tight_layout()
  plt.show()

