## Imports and parameter set up

import numpy as np
import os
import pickle
import bz2
import matplotlib.pyplot as plt
import seaborn
import string
from compute_statistics import computeOdeHigherOrderTrajectories, computeSimHigherOrderTrajectories
import variables

## Trajectories of p_t(d1, d2)
# Global Parameters
perGenReco = variables.recoTDP
thisInitScenario = 'onelow.LD'

# Constant Population
commonMu = variables.mutConstantTDP
higherOrderDemography = 'constant'

odeHigherFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'
simHigherFilename = variables.pickleDir + f'/sim.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'

if not os.path.exists(odeHigherFilename):
    computeOdeHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
if not os.path.exists(simHigherFilename):
    computeSimHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.order, variables.numReplicates, variables.numGenerations, variables.numGeneticTypes, perGenReco, commonMu, variables.pickleDir)

ifs = bz2.open (odeHigherFilename, 'rb')
odeHigherOrderData = pickle.load (ifs)
ifs.close ()

ifs = bz2.open (simHigherFilename, 'rb')
simHigherOrderData = pickle.load (ifs)
ifs.close ()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

pairs = [(0,4), (0,7), (0,8), (1,3), (4,1)]
colors = plt.cm.tab10.colors
cmap = {pair: colors[i % len(colors)] for i, pair in enumerate(pairs)}

for pair in pairs:
  color = cmap[pair]
  ax1.plot(variables.time, odeHigherOrderData[pair], label={pair}, color=color)
  ax1.plot(np.arange(variables.numGenerations), simHigherOrderData[pair], linestyle = ':', color=color)
  ax1.tick_params(labelsize=14)

ax1.set_xlabel('Generations', fontsize=14)
ax1.set_ylabel(r'$p_t^8(d^{(1)},d^{(2)})$', fontsize=14)

chandles = [ax1.plot([], [], color = cmap[pair], label={pair})[0] for pair in pairs]
handles = [ax1.plot([], [], color='black', linestyle='-', label='ODE')[0], ax1.plot([], [], color='black', linestyle=':', label='Simulation')[0]]

legend1 = ax1.legend(handles=chandles, loc='lower center', title=r'$(d^{(1)}, d^{(2)})$', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=13, title_fontsize=14)
ax1.add_artist(legend1)
legend2 = ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=14)

heatmapData = np.zeros((variables.order + 1, variables.order + 1))
for dA in np.arange(0,variables.order + 1):
    for dB in np.arange(0,variables.order + 1):
        heatmapData[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

ax2 = seaborn.heatmap(heatmapData, annot=False, cmap='Blues', square=True, vmin=0)
ax2.tick_params(labelsize=14)
ax2.set_xlabel(r'$d^{(2)}$', fontsize=14)
ax2.set_ylabel(r'$d^{(1)}$', fontsize=14)
ax2.set_title(f'u={variables.scientific_not(commonMu)}, r={variables.scientific_not(perGenReco)}', fontsize=14)
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
cbar = ax2.collections[0].colorbar
for spine in cbar.ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

plt.suptitle('Pop. Size: Constant', fontsize=18)
fig_filename = os.path.join(variables.figDir, 'constant_pTempDyn.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Bottleneck Population

commonMu = variables.mutBottleneckTDP
higherOrderDemography = 'bottleneck'

odeHigherFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'
simHigherFilename = variables.pickleDir + f'/sim.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'

if not os.path.exists(odeHigherFilename):
    computeOdeHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
if not os.path.exists(simHigherFilename):
    computeSimHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.order, variables.numReplicates, variables.numGenerations, variables.numGeneticTypes, perGenReco, commonMu, variables.pickleDir)

ifs = bz2.open (odeHigherFilename, 'rb')
odeHigherOrderData = pickle.load (ifs)
ifs.close ()

ifs = bz2.open (simHigherFilename, 'rb')
simHigherOrderData = pickle.load (ifs)
ifs.close ()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

pairs = [(0,4), (0,7), (0,8), (1,3), (4,1)]
colors = plt.cm.tab10.colors
cmap = {pair: colors[i % len(colors)] for i, pair in enumerate(pairs)}

for pair in pairs:
  color = cmap[pair]
  ax1.plot(variables.time, odeHigherOrderData[pair], label={pair}, color=color)
  ax1.plot(np.arange(variables.numGenerations), simHigherOrderData[pair], linestyle = ':', color=color)
  ax1.tick_params(labelsize=14)

ax1.set_xlabel('Generations', fontsize=14)
ax1.set_ylabel(r'$p_t^8(d^{(1)},d^{(2)})$', fontsize=14)

chandles = [ax1.plot([], [], color = cmap[pair], label={pair})[0] for pair in pairs]
handles = [ax1.plot([], [], color='black', linestyle='-', label='ODE')[0], ax1.plot([], [], color='black', linestyle=':', label='Simulation')[0]]

legend1 = ax1.legend(handles=chandles, loc='lower center', title=r'$(d^{(1)}, d^{(2)})$', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=13, title_fontsize=14)
ax1.add_artist(legend1)
legend2 = ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=14)

heatmapData = np.zeros((variables.order + 1, variables.order + 1))
for dA in np.arange(0,variables.order + 1):
    for dB in np.arange(0,variables.order + 1):
        heatmapData[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

ax2 = seaborn.heatmap(heatmapData, annot=False, cmap='Blues', square=True, vmin=0)
ax2.tick_params(labelsize=14)
ax2.set_xlabel(r'$d^{(2)}$', fontsize=14)
ax2.set_ylabel(r'$d^{(1)}$', fontsize=14)
ax2.set_title(f'u={variables.scientific_not(commonMu)}, r={variables.scientific_not(perGenReco)}', fontsize=14)
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
cbar = ax2.collections[0].colorbar
for spine in cbar.ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

plt.suptitle('Pop. Size: Bottleneck', fontsize=18)
fig_filename = os.path.join(variables.figDir, 'bottleneck_pTempDyn.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Exp. Growth Population

commonMu = variables.mutExpGrowthTDP
higherOrderDemography = 'expGrowth'

odeHigherFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'
simHigherFilename = variables.pickleDir + f'/sim.{thisInitScenario}.{higherOrderDemography}.u={commonMu}.r={perGenReco}.higherOrder={variables.order}.pkl.bz2'

if not os.path.exists(odeHigherFilename):
    computeOdeHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
if not os.path.exists(simHigherFilename):
    computeSimHigherOrderTrajectories(thisInitScenario, higherOrderDemography, variables.order, variables.numReplicates, variables.numGenerations, variables.numGeneticTypes, perGenReco, commonMu, variables.pickleDir)

ifs = bz2.open (odeHigherFilename, 'rb')
odeHigherOrderData = pickle.load (ifs)
ifs.close ()

ifs = bz2.open (simHigherFilename, 'rb')
simHigherOrderData = pickle.load (ifs)
ifs.close ()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

pairs = [(0,4), (0,7), (0,8), (1,3), (4,1)]
colors = plt.cm.tab10.colors
cmap = {pair: colors[i % len(colors)] for i, pair in enumerate(pairs)}

for pair in pairs:
  color = cmap[pair]
  ax1.plot(variables.time, odeHigherOrderData[pair], label={pair}, color=color)
  ax1.plot(np.arange(variables.numGenerations), simHigherOrderData[pair], linestyle = ':', color=color)
  ax1.tick_params(labelsize=14)

ax1.set_xlabel('Generations', fontsize=14)
ax1.set_ylabel(r'$p_t^8(d^{(1)},d^{(2)})$', fontsize=14)

chandles = [ax1.plot([], [], color = cmap[pair], label={pair})[0] for pair in pairs]
handles = [ax1.plot([], [], color='black', linestyle='-', label='ODE')[0], ax1.plot([], [], color='black', linestyle=':', label='Simulation')[0]]

legend1 = ax1.legend(handles=chandles, loc='lower center', title=r'$(d^{(1)}, d^{(2)})$', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=13, title_fontsize=14)
ax1.add_artist(legend1)
legend2 = ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=14)

heatmapData = np.zeros((variables.order + 1, variables.order + 1))
for dA in np.arange(0,variables.order + 1):
    for dB in np.arange(0,variables.order + 1):
        heatmapData[dA, dB] = odeHigherOrderData[(dA,dB)][-1]

ax2 = seaborn.heatmap(heatmapData, annot=False, cmap='Blues', square=True, vmin=0)
ax2.tick_params(labelsize=14)
ax2.set_xlabel(r'$d^{(2)}$', fontsize=14)
ax2.set_ylabel(r'$d^{(1)}$', fontsize=14)
ax2.set_title(f'u={variables.scientific_not(commonMu)}, r={variables.scientific_not(perGenReco)}', fontsize=14)
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)
cbar = ax2.collections[0].colorbar
for spine in cbar.ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.8)

plt.suptitle('Pop. Size: Exp. Growth', fontsize=18)
fig_filename = os.path.join(variables.figDir, 'expGrowth_pTempDyn.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()