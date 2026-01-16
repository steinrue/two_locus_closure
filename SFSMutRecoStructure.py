## Imports and parameter set up

import numpy as np
import os
import pickle
import bz2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn
import string
from scipy.special import rel_entr
from compute_statistics import computeOdeHigherOrderTrajectories
import variables

## Figures showing loss of independence with high vs. low r, changing shape for high vs. low u
# Low u
commonMu = variables.lowMu
scenario = 'stat'
rs = [0, 0.5]

fig, axs = plt.subplots(2, 3, figsize=(27, 16))
subfig_label = list(string.ascii_lowercase)

for i, r in enumerate(rs):
  for j, demo in enumerate(variables.demos):
    perGenReco = r
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

    if j == 0:
      vhigh = 0.055
      kws = {'ticks': [0.005, 0.02, 0.04, 0.05]}
    if j == 1:
      vhigh = 0.081
      kws = {'ticks': [0.003, 0.01, 0.02, 0.04, 0.075]}
    if j == 2:
      vhigh = 0.091
      kws = {'ticks': [0.001, 0.005, 0.02, 0.04, 0.09]}

    ax = axs[i,j]
    seaborn.heatmap(heatmapData, annot=False, cmap='Blues', cbar=True, ax=ax, norm=LogNorm(), vmin=0, vmax=vhigh, cbar_kws=kws)
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_edgecolor('black')
      spine.set_linewidth(0.8)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(True)
    for spine in cbar.ax.spines.values():
      spine.set_visible(True)
      spine.set_edgecolor('black')
      spine.set_linewidth(0.8)
    ticks = kws['ticks']
    cbar.set_ticks(ticks)
    cbar.ax.minorticks_off()
    cbar.set_ticklabels([f'{variables.scientific_not(t)}' for t in ticks])
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=30)
    if i == 1:
      ax.set_xlabel(r'$d^{(2)}$', fontsize=30)
    if j == 0:
      ax.set_ylabel(r'$d^{(1)}$', fontsize=30)
    k = i*3 + j
    ax.text(0,-0.25, f'$\\mathbf{{({subfig_label[k]})}}$ {variables.demoLabels[j]}: u={variables.scientific_not(commonMu)}, r={perGenReco}', fontsize=26)

plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'MarginalIndependenceConvex.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# High u
commonMu = variables.highMu

fig, axs = plt.subplots(2, 3, figsize=(27, 16))
subfig_label = list(string.ascii_lowercase)

for i, r in enumerate(rs):
  for j, demo in enumerate(variables.demos):
    perGenReco = r
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

    if j == 0:
      vhigh = 0.021
      kws = {'ticks': [0.005, 0.01, 0.019]}
    if j == 1:
      vhigh = 0.015
      kws = {'ticks': [0.01, 0.012, 0.0135]}
    if j == 2:
      vhigh = 0.018
      kws = {'ticks': [0.007, 0.01, 0.015]}

    ax = axs[i,j]
    seaborn.heatmap(heatmapData, annot=False, cmap='Blues', cbar=True, ax=ax, norm=LogNorm(), vmin=0, vmax=vhigh, cbar_kws=kws)
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_edgecolor('black')
      spine.set_linewidth(0.8)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(True)
    for spine in cbar.ax.spines.values():
      spine.set_visible(True)
      spine.set_edgecolor('black')
      spine.set_linewidth(0.8)
    ticks = kws['ticks']
    cbar.set_ticks(ticks)
    cbar.ax.minorticks_off()
    cbar.set_ticklabels([f'{variables.scientific_not(t)}' for t in ticks])
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=30)
    if i == 1:
      ax.set_xlabel(r'$d^{(2)}$', fontsize=30)
    if j == 0:
      ax.set_ylabel(r'$d^{(1)}$', fontsize=30)
    k = i*3 + j
    ax.text(0,-0.25, f'$\\mathbf{{({subfig_label[k]})}}$ {variables.demoLabels[j]}: u={variables.scientific_not(commonMu)}, r={perGenReco}', fontsize=26)

plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'MarginalIndependenceConcave.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Calculate and Print the MI values

for demo in variables.demos:
    commonMu = variables.lowMu
    thisInitScenario = 'stat'
    higherOrderDemography = demo

    perGenReco = rs[0]

    odeHigherFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'

    if not os.path.exists(odeHigherFilename):
        computeOdeHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)

    ifs = bz2.open (odeHigherFilename, 'rb')
    odeHigherOrderData = pickle.load (ifs)
    ifs.close ()

    jointDist = np.zeros((variables.order + 1, variables.order + 1))
    for dA in np.arange(0,variables.order + 1):
        for dB in np.arange(0,variables.order + 1):
            jointDist[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

    X1_marg = jointDist.sum(axis=1)
    X2_marg = jointDist.sum(axis=0)
    outer = np.outer(X1_marg, X2_marg)

    mimat = rel_entr(jointDist, outer)
    mi = np.sum(mimat)
    print(f'r = {perGenReco}, u = {commonMu}, {higherOrderDemography}, Mutual Information between X_1 and X_2:')
    print(mi)

    perGenReco = rs[1]

    odeHigherFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'

    if not os.path.exists(odeHigherFilename):
        computeOdeHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)

    ifs = bz2.open (odeHigherFilename, 'rb')
    odeHigherOrderData = pickle.load (ifs)
    ifs.close ()

    jointDist = np.zeros((variables.order + 1, variables.order + 1))
    for dA in np.arange(0,variables.order + 1):
        for dB in np.arange(0,variables.order + 1):
            jointDist[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

    X1_marg = jointDist.sum(axis=1)
    X2_marg = jointDist.sum(axis=0)
    outer = np.outer(X1_marg, X2_marg)

    mimat = rel_entr(jointDist, outer)
    mi = np.sum(mimat)
    print(f'r = {perGenReco}, u = {commonMu}, {higherOrderDemography}, Mutual Information between X_1 and X_2:')
    print(mi)

    commonMu = 2e-4

    perGenReco = rs[0]

    odeHigherFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'

    if not os.path.exists(odeHigherFilename):
        computeOdeHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)

    ifs = bz2.open (odeHigherFilename, 'rb')
    odeHigherOrderData = pickle.load (ifs)
    ifs.close ()

    jointDist = np.zeros((variables.order + 1, variables.order + 1))
    for dA in np.arange(0,variables.order + 1):
        for dB in np.arange(0,variables.order + 1):
            jointDist[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

    X1_marg = jointDist.sum(axis=1)
    X2_marg = jointDist.sum(axis=0)
    outer = np.outer(X1_marg, X2_marg)

    mimat = rel_entr(jointDist, outer)
    mi = np.sum(mimat)
    print(f'r = {perGenReco}, u = {commonMu}, {higherOrderDemography}, Mutual Information between X_1 and X_2:')
    print(mi)

    perGenReco = rs[1]

    odeHigherFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'

    if not os.path.exists(odeHigherFilename):
        computeOdeHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)

    ifs = bz2.open (odeHigherFilename, 'rb')
    odeHigherOrderData = pickle.load (ifs)
    ifs.close ()

    jointDist = np.zeros((variables.order + 1, variables.order + 1))
    for dA in np.arange(0,variables.order + 1):
        for dB in np.arange(0,variables.order + 1):
            jointDist[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

    X1_marg = jointDist.sum(axis=1)
    X2_marg = jointDist.sum(axis=0)
    outer = np.outer(X1_marg, X2_marg)

    mimat = rel_entr(jointDist, outer)
    mi = np.sum(mimat)
    print(f'r = {perGenReco}, u = {commonMu}, {higherOrderDemography}, Mutual Information between X_1 and X_2:')
    print(mi)