# %%
import numpy
import pickle
import bz2
from ode_solver import solveODE, getStationaryMoments
from simulated_trajectories import getSimulatedTrajectories
from HigherOrderFunctions import jointProb

# %%

# encoding: (11, 10, 01, 00)

# marginals
# 0 -- 1* = 11 + 10
# 1 -- 0* = 01 + 00
# 2 -- *1 = 11 + 01
# 3 -- *0 = 10 + 00

EPSILON = 1e-12

def getMarginals (p):
    assert (len(p.shape) == 2)
    assert (p.shape[1] == 4)

    marginals = -1 * numpy.ones (p.shape)

    # now compute
    marginals[:,0] = p[:,0] + p[:,1]
    marginals[:,1] = p[:,2] + p[:,3]
    marginals[:,2] = p[:,0] + p[:,2]
    marginals[:,3] = p[:,1] + p[:,3]

    assert (marginals.min() >= 0 - EPSILON), marginals.min()
    assert (marginals.max() <= 1 + EPSILON), marginals.max()
    marginals = numpy.clip (marginals, 0 , 1)

    return marginals

def getFreeCombinations (marginals):
    assert (len(marginals.shape) == 2)
    assert (marginals.shape[1] == 4)

    p = -1 * numpy.ones (marginals.shape)

    # now compute
    p[:,0] = marginals[:,0] * marginals[:,2]
    p[:,1] = marginals[:,0] * marginals[:,3]
    p[:,2] = marginals[:,1] * marginals[:,2]
    p[:,3] = marginals[:,1] * marginals[:,3]

    assert (p.min() >= 0 - EPSILON), p.min()
    assert (p.max() <= 1 + EPSILON), p.max()
    p = numpy.clip (p, 0 , 1)

    return p

def LD (p):
    return p - getFreeCombinations (getMarginals (p))


# %%

def computeODEStatistics(thisInitScenario, thisPopScenario, initialStates, order, time, perGenReco, mu, WD):

    G, stateToIdx = solveODE(thisInitScenario, thisPopScenario, initialStates, order, time, perGenReco, mu)

    ED2=G[stateToIdx[(2,0,0)]]-2*G[stateToIdx[(1,1,1)]]+G[stateToIdx[(0,2,2)]]
    ED=G[stateToIdx[(1,0,0)]]-G[stateToIdx[(0,1,1)]]
    EX1 = G[stateToIdx[(0,1,0)]]
    EX2 = G[stateToIdx[(0,0,1)]]
    Ehet1 = 2*G[stateToIdx[(0,1,0)]] - 2*G[stateToIdx[(0,2,0)]]
    Ehet2 = 2*G[stateToIdx[(0,0,1)]] - 2*G[stateToIdx[(0,0,2)]]

    odeData = {'meanOne' : EX1,
    'meanTwo' : EX2,
    'meanHetOne' : Ehet1,
    'meanHetTwo' : Ehet2,
    'meanLD' : ED,
    'meanLDSQ' : ED2}

    filename = WD + f'/ode.{thisInitScenario}.{thisPopScenario}.u={mu}.r={perGenReco}.pkl.bz2'

    ofs = bz2.open (filename, 'wb')
    pickle.dump (odeData, ofs)
    ofs.close()

# %%

def computeStationaryStatistics(thisPopScenario, initialStates, order, perGenReco, mu, WD):

    stationaryG, stateToIdx = getStationaryMoments(thisPopScenario, initialStates, order, perGenReco, mu)

    ED2stat = stationaryG[stateToIdx[(2,0,0)] - 1]-2*stationaryG[stateToIdx[(1,1,1)]- 1]+stationaryG[stateToIdx[(0,2,2)] - 1]
    EDstat=stationaryG[stateToIdx[(1,0,0)] - 1]-stationaryG[stateToIdx[(0,1,1)] - 1]
    EX1stat = stationaryG[stateToIdx[(0,1,0)] - 1]
    EX2stat = stationaryG[stateToIdx[(0,0,1)] - 1]
    Ehet1stat = 2*stationaryG[stateToIdx[(0,1,0)] - 1] - 2*stationaryG[stateToIdx[(0,2,0)] - 1]
    Ehet2stat = 2*stationaryG[stateToIdx[(0,0,1)] - 1] - 2*stationaryG[stateToIdx[(0,0,2)] - 1]

    statData = {'One' : EX1stat,
    'Two' : EX2stat,
    'HetOne' : Ehet1stat,
    'HetTwo' : Ehet2stat,
    'LD' : EDstat,
    'LDSQ' : ED2stat}

    filename = WD + f'/stat.{thisPopScenario}.u={mu}.r={perGenReco}.pkl.bz2'

    ofs = bz2.open (filename, 'wb')
    pickle.dump (statData, ofs)
    ofs.close()

# %%

def computeSimulatedStatistics(thisInitScenario, thisPopScenario, numReplicates, numGenerations, numGeneticTypes, rho, commonMu, WD):

    simulatedTrajectories = getSimulatedTrajectories(thisInitScenario, thisPopScenario, numReplicates, numGenerations, numGeneticTypes, rho, commonMu)

    # first locus
    # X1
    margOne = simulatedTrajectories[:,:,0] + simulatedTrajectories[:,:,1]
    meanOne = numpy.mean (margOne, axis=0)
    sdOne = numpy.std (margOne, axis=0)

    # 2*X1*(1-X1)
    hetOne = 2 * margOne * (1-margOne)
    meanHetOne = numpy.mean (hetOne, axis=0)
    sdHetOne = numpy.std (hetOne, axis=0)

    # second locus
    # X2
    margTwo = simulatedTrajectories[:,:,0] + simulatedTrajectories[:,:,2]
    meanTwo = numpy.mean (margTwo, axis=0)
    sdTwo = numpy.std (margTwo, axis=0)

    # 2*X2*(1-X2)
    hetTwo = 2 * margTwo * (1-margTwo)
    meanHetTwo = numpy.mean (hetTwo, axis=0)
    sdHetTwo = numpy.std (hetTwo, axis=0)

    # LD stuff
    allLD = numpy.zeros ((simulatedTrajectories.shape[0], simulatedTrajectories.shape[1]))
    for genIdx in numpy.arange(simulatedTrajectories.shape[1]):
        allLD[:,genIdx] = LD(simulatedTrajectories[:,genIdx,:])[:,0]
    meanLD = numpy.mean (allLD, axis=0)
    sdLD = numpy.std (allLD, axis=0)

    # LD^2 stuff
    allLDSQ = numpy.zeros ((simulatedTrajectories.shape[0], simulatedTrajectories.shape[1]))
    for genIdx in numpy.arange(simulatedTrajectories.shape[1]):
        allLDSQ[:,genIdx] = LD(simulatedTrajectories[:,genIdx,:])[:,0]
        allLDSQ[:,genIdx] = allLDSQ[:,genIdx] * allLDSQ[:,genIdx]
    meanLDSQ = numpy.mean (allLDSQ, axis=0)
    sdLDSQ = numpy.std (allLDSQ, axis=0)

    simulatedData = {'meanOne' : meanOne,
    'sdOne' : sdOne,
    'meanTwo' : meanTwo,
    'sdTwo' : sdTwo,
    'meanHetOne' : meanHetOne,
    'sdHetOne' : sdHetOne,
    'meanHetTwo' : meanHetTwo,
    'sdHetTwo' : sdHetTwo,
    'meanLD' : meanLD,
    'sdLD' : sdLD,
    'meanLDSQ' : meanLDSQ,
    'sdLDSQ' : sdLDSQ}

    filename = WD + f'/sim.{thisInitScenario}.{thisPopScenario}.u={commonMu}.r={rho}.pkl.bz2'

    ofs = bz2.open (filename, 'wb')
    pickle.dump (simulatedData, ofs)
    ofs.close()



# %%

# multinomial function (copied from stack exchange)
from scipy.special import binom

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


# %%

def computeOdeHigherOrderTrajectories(thisInitScenario, thisPopScenario, initialStates, order, time, perGenReco, mu, WD):

    G, stateToIdx = solveODE(thisInitScenario, thisPopScenario, initialStates, order, time, perGenReco, mu)

    odeHigherOrderTrajectories = {}

    for dA in numpy.arange(0,order + 1):
        for dB in numpy.arange(0,order + 1):
            odeHigherOrderTrajectories[(dA,dB)] = jointProb(dA,dB, order, G, stateToIdx)

    
    filename = WD + f'/ode.{thisInitScenario}.{thisPopScenario}.u={mu}.r={perGenReco}.higherOrder={order}.pkl.bz2'
    
    ofs = bz2.open (filename, 'wb')
    pickle.dump (odeHigherOrderTrajectories, ofs)
    ofs.close()


# %%

def computeSimHigherOrderTrajectories(thisInitScenario, thisPopScenario, order, numReplicates, numGenerations, numGeneticTypes, rho, commonMu, WD):
    
    simulatedTrajectories = getSimulatedTrajectories(thisInitScenario, thisPopScenario, numReplicates, numGenerations, numGeneticTypes, rho, commonMu)

    simulatedHigherOrderTrajectories = {}


    for dA in numpy.arange(0,order + 1):
        for dB in numpy.arange(0,order + 1):
            simulatedHigherOrderTrajectories[(dA,dB)] = numpy.mean(sum(multinomial([nAB, dA - nAB, dB - nAB, order + nAB - dA - dB]) * simulatedTrajectories[:,:,0]**(nAB) * simulatedTrajectories[:,:,1]**(dA - nAB) *
                    simulatedTrajectories[:,:,2]**(dB - nAB) * simulatedTrajectories[:,:,3]**(order + nAB - dA - dB)  
                    for nAB in range(max(0,dA + dB - order) , min(dA,dB) + 1)), axis = 0)
            
    
    
    filename = WD + f'/sim.{thisInitScenario}.{thisPopScenario}.u={commonMu}.r={rho}.higherOrder={order}.pkl.bz2'
    
    ofs = bz2.open (filename, 'wb')
    pickle.dump (simulatedHigherOrderTrajectories, ofs)
    ofs.close()
