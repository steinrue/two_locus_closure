## Imports and parameter set up

import numpy as np
import os
import pickle
import bz2
import matplotlib.pyplot as plt
from compute_statistics import computeODEStatistics, computeStationaryStatistics, computeSimulatedStatistics
import variables

# Trajectories of Statistics of Interest
# Constant Population

perGenReco = variables.recoConstantBottleneckStatTraj
# High mutation rate so we observe quicker rate of convergence over 2000 generations
commonMu = variables.mutConstantStatTraj
demography = 'constant'

stats = ['One', 'Two', 'HetOne', 'HetTwo', 'LD', 'LDSQ']
statLabels = [r'$\mathbf{E}(X_{1\cdot}(t))$', r'$\mathbf{E}(X_{\cdot 1}(t))$', r'$\mathbf{H_1}(t)$', r'$\mathbf{H_2}(t)$', r'$\mathbf{D}(t)$', r'$\mathbf{D}^2(t)$']
ICs = ['(0.5, 0.5) - w/o LD', '(0.5, 0.5) - w/ LD', '(0.05, 0.5) - w/o LD', '(0.05, 0.5) - w/ LD', '(0.05, 0.05) - w/o LD', '(0.05, 0.05) - w/ LD']
# restrict which ICs are shown for marginal frequency and expected heterozygosity
subset_IC_ids = [0, 5]
# symmetric log plot threshold value
symLogThresh = 1e-4

f, axs = plt.subplots(2,3, figsize=(7.5, 4))
colors = plt.cm.tab10.colors

counter = 0
for j in np.arange(3):
    for i in np.arange(2):

        thisStat = stats[counter]
        ax = axs[i,j]
        ax.axhline(0, color='black', linewidth=0.5, linestyle='dashed')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='dashed')

        sim_plot_data = []

        for idx, thisInitScenario in enumerate(variables.scenarios):

            # restrict which ICs are shown for marginal frequency and expected heterozygosity
            if j != 2 and idx not in subset_IC_ids:
                continue
            # include only nonzero LD figures
            if j == 2 and i == 0 and idx not in [1, 3, 5]:
                continue

            odeFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{demography}.u={commonMu}.r={perGenReco}.pkl.bz2'
            simFilename = variables.pickleDir + f'/sim.{thisInitScenario}.{demography}.u={commonMu}.r={perGenReco}.pkl.bz2'

            if not os.path.exists(odeFilename):
                 computeODEStatistics(thisInitScenario, demography , variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
            if not os.path.exists(simFilename):
                 computeSimulatedStatistics(thisInitScenario, demography, variables.numReplicates, variables.numGenerations, variables.numGeneticTypes, perGenReco, commonMu, variables.pickleDir)

            ifs = bz2.open (odeFilename, 'rb')
            odeData = pickle.load (ifs)
            ifs.close ()

            ifs = bz2.open (simFilename, 'rb')
            simulatedData = pickle.load (ifs)
            ifs.close ()
            ode_plot, = ax.plot(variables.time, odeData[f'mean{thisStat}'], color=colors[idx], label=f"{thisInitScenario} (ODE)")
            # plot subsample of simulated trajectory for visibility
            sim_time = np.arange(variables.numGenerations)
            sim_vals = simulatedData[f'mean{thisStat}']
            num_subsample = 11
            # set up so crosses are staggered
            base_locs = np.linspace(0, variables.numGenerations - 1, num_subsample)
            width = variables.numGenerations / (num_subsample*len(variables.scenarios))
            stagger = idx*width
            subsample_idx = np.round((base_locs+stagger) % variables.numGenerations).astype(int)
            subsample_idx = np.sort(subsample_idx)
            sim_plot_data.append((idx, thisInitScenario, sim_time[subsample_idx], sim_vals[subsample_idx]))

        # plot all simulated trajectories after all ODE trajectories
        for idx, thisInitScenario, sim_t, sim_v in sim_plot_data:
            sim_plot, = ax.plot(sim_t, sim_v, marker='x', linestyle='None', color=colors[idx], label=f'{thisInitScenario} (Sim)')

        statFilename = variables.pickleDir + f'/stat.{demography}.u={commonMu}.r={perGenReco}.pkl.bz2'
        if not os.path.exists(statFilename):
            computeStationaryStatistics(demography, variables.initialStates, variables.order, perGenReco, commonMu, variables.pickleDir)
        ifs = bz2.open (statFilename, 'rb')
        stationaryData = pickle.load (ifs)
        ifs.close ()
        if thisStat != 'LD':
            ax.hlines(stationaryData[thisStat], 0, variables.numGenerations, linestyles='dashed', colors='black')

        ax.set_xlabel(statLabels[counter])
        if thisStat == 'LD':
            # log scale with floor at 1e-4
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-4)
        elif thisStat == 'LDSQ':
            # symmetric log scale
            ax.set_yscale('symlog', linthresh=1e-4, linscale=0.3)
            # delineation between linear and log scale regions
            ax.axhspan(0, symLogThresh, color='lightgray', alpha=0.3, zorder=0)
        else:
            ax.set_ylim(-0.035, 0.65)
        counter += 1

scenario_handles = [axs[0,0].plot([], [], color=colors[id], label=scenario)[0] for id, scenario in enumerate(ICs)]
method_handles = [axs[0,0].plot([], [], color='black', linestyle='-', label='ODE')[0], axs[0,0].plot([], [], color='black', marker='x', linestyle='None', label='Sim')[0], axs[0,0].plot([], [], color='black', linestyle='dashed', label='Stationary Value')[0]]
legend1 = f.legend(handles=scenario_handles, loc='lower center', title=r'$(X_{1\cdot}(0), X_{\cdot 1}(0))$', bbox_to_anchor=(0.5, -0.16), ncol=3)
f.add_artist(legend1)
legend2 = f.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=3)

plt.suptitle('Pop. Size: Constant', fontsize=11)
plt.subplots_adjust(wspace=0.2, hspace=0.425)
plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'ConstantStatTraj.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Bottleneck Population

commonMu = variables.mutBottleneckStatTraj
demography = 'bottleneck'

f, axs = plt.subplots(2,3, figsize=(7.5, 4))
colors = plt.cm.tab10.colors

counter = 0
for j in np.arange(3):
    for i in np.arange(2):

        thisStat = stats[counter]
        ax = axs[i,j]
        ax.axhline(0, color='black', linewidth=0.5, linestyle='dashed')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='dashed')

        sim_plot_data = []

        for idx, thisInitScenario in enumerate(variables.scenarios):

            # restrict which ICs are shown for marginal frequency and expected heterozygosity
            if j != 2 and idx not in subset_IC_ids:
                continue
            # include only nonzero LD figures
            if j == 2 and i == 0 and idx not in [1, 3, 5]:
                continue

            odeFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{demography}.u={commonMu}.r={perGenReco}.pkl.bz2'
            simFilename = variables.pickleDir + f'/sim.{thisInitScenario}.{demography}.u={commonMu}.r={perGenReco}.pkl.bz2'

            if not os.path.exists(odeFilename):
                 computeODEStatistics(thisInitScenario, demography , variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
            if not os.path.exists(simFilename):
                 computeSimulatedStatistics(thisInitScenario, demography, variables.numReplicates, variables.numGenerations, variables.numGeneticTypes, perGenReco, commonMu, variables.pickleDir)

            ifs = bz2.open (odeFilename, 'rb')
            odeData = pickle.load (ifs)
            ifs.close ()

            ifs = bz2.open (simFilename, 'rb')
            simulatedData = pickle.load (ifs)
            ifs.close ()
            ode_plot, = ax.plot(variables.time, odeData[f'mean{thisStat}'], color=colors[idx], label=f"{thisInitScenario} (ODE)")
            # plot subsample of simulated trajectory for visibility
            sim_time = np.arange(variables.numGenerations)
            sim_vals = simulatedData[f'mean{thisStat}']
            num_subsample = 11
            # set up so crosses are staggered
            base_locs = np.linspace(0, variables.numGenerations - 1, num_subsample)
            width = variables.numGenerations / (num_subsample*len(variables.scenarios))
            stagger = idx*width
            subsample_idx = np.round((base_locs+stagger) % variables.numGenerations).astype(int)
            subsample_idx = np.sort(subsample_idx)
            sim_plot_data.append((idx, thisInitScenario, sim_time[subsample_idx], sim_vals[subsample_idx]))

        # plot all simulated trajectories after all ODE trajectories
        for idx, thisInitScenario, sim_t, sim_v in sim_plot_data:
            sim_plot, = ax.plot(sim_t, sim_v, marker='x', linestyle='None', color=colors[idx], label=f'{thisInitScenario} (Sim)')

        ax.set_xlabel(statLabels[counter])
        if thisStat == 'LD':
            # log scale with floor at 1e-4
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-4)
        elif thisStat == 'LDSQ':
            # symmetric log scale
            ax.set_yscale('symlog', linthresh=1e-4, linscale=0.3)
            # delineation between linear and log scale regions
            ax.axhspan(0, symLogThresh, color='lightgray', alpha=0.3, zorder=0)
        else:
            ax.set_ylim(-0.035, 0.65)
        counter += 1

scenario_handles = [axs[0,0].plot([], [], color=colors[id], label=scenario)[0] for id, scenario in enumerate(ICs)]
method_handles = [axs[0,0].plot([], [], color='black', linestyle='-', label='ODE')[0], axs[0,0].plot([], [], color='black', marker='x', linestyle='None', label='Sim')[0]]
legend1 = f.legend(handles=scenario_handles, loc='lower center', title=r'$(X_{1\cdot}(0), X_{\cdot 1}(0))$', bbox_to_anchor=(0.5, -0.16), ncol=3)
f.add_artist(legend1)
legend2 = f.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=2)

plt.suptitle('Pop. Size: Bottleneck', fontsize=11)
plt.subplots_adjust(wspace=0.2, hspace=0.425)
plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'BottleneckStatTraj.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()

# Exp. Growth

perGenReco = variables.recoExpGrowthStatTraj
commonMu = variables.mutExpGrowthStatTraj
demography = 'expGrowth'

f, axs = plt.subplots(2,3, figsize=(7.5, 4))
colors = plt.cm.tab10.colors

counter = 0
for j in np.arange(3):
    for i in np.arange(2):

        thisStat = stats[counter]
        ax = axs[i,j]
        ax.axhline(0, color='black', linewidth=0.5, linestyle='dashed')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='dashed')

        sim_plot_data = []

        for idx, thisInitScenario in enumerate(variables.scenarios):

            # restrict which ICs are shown for marginal frequency and expected heterozygosity
            if j != 2 and idx not in subset_IC_ids:
                continue
            # include only nonzero LD figures
            if j == 2 and i == 0 and idx not in [1, 3, 5]:
                continue

            odeFilename = variables.pickleDir + f'/ode.{thisInitScenario}.{demography}.u={commonMu}.r={perGenReco}.pkl.bz2'
            simFilename = variables.pickleDir + f'/sim.{thisInitScenario}.{demography}.u={commonMu}.r={perGenReco}.pkl.bz2'

            if not os.path.exists(odeFilename):
                 computeODEStatistics(thisInitScenario, demography , variables.initialStates, variables.order, variables.time, perGenReco, commonMu, variables.pickleDir)
            if not os.path.exists(simFilename):
                 computeSimulatedStatistics(thisInitScenario, demography, variables.numReplicates, variables.numGenerations, variables.numGeneticTypes, perGenReco, commonMu, variables.pickleDir)

            ifs = bz2.open (odeFilename, 'rb')
            odeData = pickle.load (ifs)
            ifs.close ()

            ifs = bz2.open (simFilename, 'rb')
            simulatedData = pickle.load (ifs)
            ifs.close ()
            ode_plot, = ax.plot(variables.time, odeData[f'mean{thisStat}'], color=colors[idx], label=f"{thisInitScenario} (ODE)")
            # plot subsample of simulated trajectory for visibility
            sim_time = np.arange(variables.numGenerations)
            sim_vals = simulatedData[f'mean{thisStat}']
            num_subsample = 11
            # set up so crosses are staggered
            base_locs = np.linspace(0, variables.numGenerations - 1, num_subsample)
            width = variables.numGenerations / (num_subsample*len(variables.scenarios))
            stagger = idx*width
            subsample_idx = np.round((base_locs+stagger) % variables.numGenerations).astype(int)
            subsample_idx = np.sort(subsample_idx)
            sim_plot_data.append((idx, thisInitScenario, sim_time[subsample_idx], sim_vals[subsample_idx]))

        # plot all simulated trajectories after all ODE trajectories
        for idx, thisInitScenario, sim_t, sim_v in sim_plot_data:
            sim_plot, = ax.plot(sim_t, sim_v, marker='x', linestyle='None', color=colors[idx], label=f'{thisInitScenario} (Sim)')

        ax.set_xlabel(statLabels[counter])
        if thisStat == 'LD':
            # log scale with floor at 1e-4
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-4)
        elif thisStat == 'LDSQ':
            # symmetric log scale
            ax.set_yscale('symlog', linthresh=1e-4, linscale=0.3)
            # delineation between linear and log scale regions
            ax.axhspan(0, symLogThresh, color='lightgray', alpha=0.3, zorder=0)
        else:
            ax.set_ylim(-0.035, 0.65)
        counter += 1

scenario_handles = [axs[0,0].plot([], [], color=colors[id], label=scenario)[0] for id, scenario in enumerate(ICs)]
method_handles = [axs[0,0].plot([], [], color='black', linestyle='-', label='ODE')[0], axs[0,0].plot([], [], color='black', marker='x', linestyle='None', label='Sim')[0]]
legend1 = f.legend(handles=scenario_handles, loc='lower center', title=r'$(X_{1\cdot}(0), X_{\cdot1}(0))$', bbox_to_anchor=(0.5, -0.16), ncol=3)
f.add_artist(legend1)
legend2 = f.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(0.5, -0.23), ncol=2)

plt.suptitle('Pop. Size: Exp. Growth', fontsize=11)
plt.subplots_adjust(wspace=0.2, hspace=0.425)
plt.tight_layout()
fig_filename = os.path.join(variables.figDir, 'ExpGrowthStatTraj.pdf')
plt.savefig(fig_filename, format='pdf', bbox_inches='tight')
# plt.show()