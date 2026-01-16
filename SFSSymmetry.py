## Imports and parameter set up

import numpy as np
import os
import pickle
import bz2
import matplotlib.pyplot as plt
import seaborn
import string
from compute_statistics import computeODEStatistics, computeStationaryStatistics, computeSimulatedStatistics, computeOdeHigherOrderTrajectories, computeSimHigherOrderTrajectories
import variables

## Symmetry of haplotype frequency matrix at stationarity
# Parameters
perGenReco = variables.recoSFS
commonMu = variables.mutSFS
ICs = ['stat', 'onelow.LD']

fig, axs = plt.subplots(2, 3, figsize=(27, 18))
subfig_label = list(string.ascii_lowercase)

for i, scenario in enumerate(ICs):
  for j, demo in enumerate(variables.demos):
    odeHigherFilename = variables.pickleDir + f'/ode.{scenario}.{demo}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'
    if not os.path.exists(odeHigherFilename):
      computeOdeHigherOrderTrajectories(scenario, demo, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
    ifs = bz2.open (odeHigherFilename, 'rb')
    odeHigherOrderData = pickle.load (ifs)
    ifs.close ()
    heatmapData = np.zeros((variables.order + 1, variables.order + 1))
    for dA in np.arange(0,variables.order + 1):
      for dB in np.arange(0,variables.order + 1):
        heatmapData[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

    ax = axs[i,j]
    if i == 0:
      seaborn.heatmap(heatmapData, annot=False, cmap='Blues', cbar=False, vmin=0, vmax=0.025, ax=ax)
    if i == 1:
      seaborn.heatmap(heatmapData, annot=False, cmap='Blues', cbar=False, vmin=0, vmax=0.03, ax=ax)
      ax.set_xlabel(r'$d^{(2)}$', fontsize=30)
    if j == 0:
      ax.set_ylabel(r'$d^{(1)}$', fontsize=30)
    ax.tick_params(labelsize=30)
    k = i*3 + j
    ax.text(0,-0.25, f'$\\mathbf{{({subfig_label[k]})}}$ {variables.demoLabels[j]}: u={variables.scientific_not(commonMu)}, r={variables.scientific_not(perGenReco)}', fontsize=26)
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_edgecolor('black')
      spine.set_linewidth(0.8)

  cbar_ax = fig.add_axes([1, 0.55 - i*0.475, 0.02, 0.4])
  fig.colorbar(axs[i, 0].collections[0], cax=cbar_ax)
  cbar_ax.tick_params(labelsize=20)

plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'SFSSymmetry.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
plt.show()