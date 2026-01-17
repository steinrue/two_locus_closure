import os
import numpy as np
from scipy.spatial import distance

# Locations to save figures and pickle files
parentDir = os.path.dirname(os.path.abspath(__file__))
figDir = os.path.join(parentDir, 'figures')
pickleDir = os.path.join(parentDir, 'pickles')
os.makedirs(figDir, exist_ok=True)
os.makedirs(pickleDir, exist_ok=True)

# ODE and Simulation parameters
initialStates = [(2,0,0), (0,2,0), (0,0,2), (2,2,2)]
order = 8
time = np.linspace(0,2000,200)
numGenerations = 2000
numReplicates = 8192
numGeneticTypes = 4

# IC Lists
scenarios = ['nolow.noLD', 'nolow.LD', 'onelow.noLD', 'onelow.LD', 'bothlow.noLD', 'bothlow.LD']
demos = ['constant', 'bottleneck', 'expGrowth']
demoLabels = ['Constant', 'Bottleneck', 'Exp. Growth']
# Definition of IC values in a dictionary
allScenarios = {
    'nolow.noLD' : {'x0' : [0.25, 0.25, 0.25, 0.25]},
    'nolow.LD' : {'x0' : [0.4, 0.1, 0.1, 0.4]},
    'onelow.noLD' : {'x0' : [0.025, 0.025, 0.475, 0.475]},
    'onelow.LD' : {'x0' : [0.04, 0.01, 0.46, 0.49]},
    'bothlow.noLD' : {'x0' : [0.0025, 0.0475, 0.0475, 0.9025]},
    'bothlow.LD' : {'x0' : [0.00375, 0.04625, 0.04625, 0.90375]}
}
# Population size parameters to generate msprime demographies and ODE/Simulation populations
constantSize = 2000
bottleneckFull = 2000
bottleneckTimeFull = 750
bottleneckShrink = 500
bottleneckTimeShrink = 500
expGrowthStart = 500
expFinalSize = 40000
expGrowthTime = 1000
expGrowthRate = np.log(80)/1000
# Creates population sizes for ODE and Simulation
popScenarios = {
    'constant' : np.full((numGenerations),constantSize,dtype=int),
    'bottleneck' : np.concatenate((np.full((bottleneckTimeFull),bottleneckFull), np.full((bottleneckTimeShrink),bottleneckShrink), np.full((bottleneckTimeFull),bottleneckFull)), dtype=int),
    'expGrowth' : np.concatenate((np.full((1,expGrowthTime),expGrowthStart), np.int64(np.round(expGrowthStart * np.exp((expGrowthRate)*(np.arange(expGrowthTime,numGenerations,1)-expGrowthTime))))),axis = None, dtype=int),
}
# msprime Simulation parameters - number of replicates
pairwiseReplicates = int(2**20)
ploidy = 8
msprimeMuts = [2e-4]
pairwiseReco = [1e-4]
genomeReplicates = 64
seqLength = int(4e4)
genomicReco = 1e-6
recoMids = np.array([100, 400])
pm = 25

# Helper function for scientific notation
def scientific_not(num):
  exp_string = f'{num:.2e}'
  coef, exp = exp_string.split('e')
  exp = int(exp)
  coef = coef.rstrip('0')
  if coef[-1] == '.':
    coef = coef[:-1]
  return fr'${coef} \cdot 10^{{{exp}}}$'

# Helper function for JS divergence between haplotype frequency matrices
def discreteJSD(P, Q):
  P_1D = np.ravel(P)
  Q_1D = np.ravel(Q)
  jsd = distance.jensenshannon(P_1D, Q_1D, base=2)**2
  return jsd

# Helper function for TV distance between haplotype frequency matrices
def TVD(P, Q):
  abs_diff = np.abs(P - Q)
  tvd = 0.5*np.sum(abs_diff)
  return tvd

# Rates for figures
# Statistic Trajectories
recoConstantBottleneckStatTraj = 1e-4
mutConstantStatTraj = 5e-4
mutBottleneckStatTraj = 2e-4
recoExpGrowthStatTraj = 5e-4
mutExpGrowthStatTraj = 1e-4
# Temporal Dynamics of p(d1, d2)
recoTDP = 1e-4
mutConstantTDP = 1e-4
mutBottleneckTDP = 2e-4
mutExpGrowthTDP = 5e-5
# ODE vs. Msprime and SFS Symmetric
recoSFS = 1e-4
mutSFS = 2e-4
# SFS Structure for Low and High u and r
lowMu = 5e-5
highMu = 2e-4