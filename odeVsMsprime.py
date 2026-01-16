## Imports, path and parameter set up

import numpy as np
import os
import pickle
import bz2
import matplotlib.pyplot as plt
import seaborn
import string
from compute_statistics import computeOdeHigherOrderTrajectories
import variables

## Comparison of ODE and msprime haplotype frequency matrices

# Global parameters for this figure
perGenReco = variables.recoSFS
commonMu = variables.mutSFS

fig, axs = plt.subplots(3, 3, figsize=(27,27))
subfig_label = list(string.ascii_lowercase)
odeDict = {}

# load pickle files
with bz2.open('pickles/msprime_simulations.pickle.bz2', 'rb') as f:
  msprime_data = pickle.load(f)

pairwise_jSFSs = msprime_data['pairwiseJSFSs']
pairwise_relErrs = msprime_data['pairwiseRelErr']
genomic_jSFSs = msprime_data['genomicJSFSs']
genomic_relErrs = msprime_data['genomicRelErr']
mutRecoLabel = 'm_2.00000e-04_r1.00000e-04'

for i, demo in enumerate(variables.demos):
  odeHigherFilename = variables.pickleDir + f'/ode.stat.{demo}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'
  if not os.path.exists(odeHigherFilename):
    computeOdeHigherOrderTrajectories('stat', demo, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
  ifs = bz2.open(odeHigherFilename, 'rb')
  odeHigherOrderData = pickle.load(ifs)
  ifs.close()
  heatmapData = np.zeros((variables.order + 1, variables.order + 1))
  for dA in np.arange(0, variables.order + 1):
    for dB in np.arange(0, variables.order + 1):
      heatmapData[dA, dB] = odeHigherOrderData[(dA,dB)][-1]
  
  odeDict[demo] = heatmapData

  if i == 0:
    vhigh = 0.026
  else:
    vhigh = 0.02

  ax = axs[i,0]
  seaborn.heatmap(heatmapData, annot=False, vmin=0, vmax=vhigh, cmap='Blues', cbar=False, ax=ax)
  ax.set_ylabel(r'$d^{(1)}$', fontsize=30)
  if i == 2:
    ax.set_xlabel(r'$d^{(2)}$', fontsize=30)
  ax.tick_params(labelsize=30)
  k = i*3
  ax.text(0,-0.25, f'$\\mathbf{{({subfig_label[k]})}}$ {variables.demoLabels[i]}: u={variables.scientific_not(commonMu)}, r={variables.scientific_not(perGenReco)}', fontsize=26)
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
  k += 1

  ax = axs[i, 1]
  SFSmat = pairwise_jSFSs[demo][mutRecoLabel]
  seaborn.heatmap(SFSmat, annot=False, vmin=0, vmax=vhigh, cmap='Blues', cbar=False, ax=ax)
  if i == 2:
    ax.set_xlabel(r'$d^{(2)}$', fontsize=30)
  ax.tick_params(labelsize=30)
  ax.text(0, -0.25, f'$\\mathbf{{({subfig_label[k]})}}$ {variables.demoLabels[i]}: $M_p$', fontsize=26)
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
  k += 1

  ax = axs[i, 2]
  SFSmat = genomic_jSFSs[demo][mutRecoLabel]
  seaborn.heatmap(SFSmat, annot=False, vmin=0, vmax=vhigh, cmap='Blues', cbar=False, ax=ax)
  if i == 2:
    ax.set_xlabel(r'$d^{(2)}$', fontsize=30)
  ax.tick_params(labelsize=30)
  ax.text(0, -0.25, f'$\\mathbf{{({subfig_label[k]})}}$ {variables.demoLabels[i]}: $M_g$', fontsize=26)
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
  k += 1

  cbar_ax = fig.add_axes([1, 0.695-0.327*i, 0.02, 0.25])
  fig.colorbar(axs[i, 0].collections[0], cax=cbar_ax)
  cbar_ax.tick_params(labelsize=20)

plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'ODEvsMsprime.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Plot relative error for each bin, trating the ODE as 'truth'
fig, axs = plt.subplots(2, 3, figsize=(27, 18))
for i, demo in enumerate(variables.demos):
  odeHigherFilename = variables.pickleDir + f'/ode.stat.{demo}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'
  if not os.path.exists(odeHigherFilename):
    computeOdeHigherOrderTrajectories('stat', demo, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
  ifs = bz2.open(odeHigherFilename, 'rb')
  odeHigherOrderData = pickle.load(ifs)
  ifs.close()
  heatmapData = np.zeros((variables.order + 1, variables.order + 1))
  for dA in np.arange(0, variables.order + 1):
    for dB in np.arange(0, variables.order + 1):
      heatmapData[dA, dB] = odeHigherOrderData[(dA,dB)][-1]
  
  MpRelErr = np.divide(np.abs(pairwise_jSFSs[demo][mutRecoLabel] - heatmapData), np.abs(heatmapData))
  seaborn.heatmap(MpRelErr, annot=True, vmin=0, cmap='rocket_r', cbar=False, ax=axs[0, i])
  axs[0, i].text(0, -0.25, fr'{variables.demoLabels[i]}: $M_p$ Relative Error (ODE Truth)')

  MgRelErr = np.divide(np.abs(genomic_jSFSs[demo][mutRecoLabel] - heatmapData), np.abs(heatmapData))
  seaborn.heatmap(MgRelErr, annot=True, vmin=0, cmap='rocket_r', cbar=False, ax=axs[1, i])
plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'RelErrorODETruth.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Plot monte carlo errors in simulation for each demography
fig, axs = plt.subplots(2, 3, figsize=(27, 18))
for i, demo in enumerate(variables.demos):
  seaborn.heatmap(pairwise_relErrs[demo][mutRecoLabel], annot=True, vmin=0, cmap='rocket_r', cbar=False, ax=axs[0, i])
  axs[0, i].text(0, -0.25, fr'{variables.demoLabels[i]}: $M_p$ Monte Carlo Error', fontsize=26)

  seaborn.heatmap(genomic_relErrs[demo][mutRecoLabel], annot=True, vmin=0, cmap='rocket_r', cbar=False, ax=axs[1, i])
  axs[1, i].text(0, -0.25, fr'{variables.demoLabels[i]}: $M_g$ Monte Carlo Error', fontsize=26)
plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'MonteCarloError.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Compute JS Divergence and TV Distance Between pairs of heatmaps for each demo
for demo in variables.demos:
  P = odeDict[demo]
  Q = pairwise_jSFSs[demo][mutRecoLabel]
  R = genomic_jSFSs[demo][mutRecoLabel]
  tvd = variables.TVD(P, Q)
  jsd = variables.discreteJSD(P, Q)
  print(f'TVD Between ODE and $M_p$ for {demo} - {variables.scientific_not(tvd)}')
  print(f'JSD Between ODE and $M_p$ for {demo} - {variables.scientific_not(jsd)}')
  tvd = variables.TVD(P, R)
  jsd = variables.discreteJSD(P, R)
  print(f'TVD Between ODE and $M_g$ for {demo} - {variables.scientific_not(tvd)}')
  print(f'JSD Between ODE and $M_g$ for {demo} - {variables.scientific_not(jsd)}')
  tvd = variables.TVD(Q, R)
  jsd = variables.discreteJSD(Q, R)
  print(f'TVD Between $M_p$ and $M_g$ for {demo} - {variables.scientific_not(tvd)}')
  print(f'JSD Between $M_p$ and $M_g$ for {demo} - {variables.scientific_not(jsd)}')