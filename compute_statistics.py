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

    # chunk numReplicates into batches to be able to print progress update and prevent memory issues
    numBatches = 25
    batchSize = max(1, numReplicates//numBatches)

    sumOne = numpy.zeros(numGenerations)
    sumOneSq = numpy.zeros(numGenerations)
    sumHetOne = numpy.zeros(numGenerations)
    sumHetOneSq = numpy.zeros(numGenerations)
    sumTwo = numpy.zeros(numGenerations)
    sumTwoSq = numpy.zeros(numGenerations)
    sumHetTwo = numpy.zeros(numGenerations)
    sumHetTwoSq = numpy.zeros(numGenerations)
    # required values to store to compute mean and sd of D and D^2
    sumD = numpy.zeros(numGenerations)
    sumD2 = numpy.zeros(numGenerations)
    sumD4 = numpy.zeros(numGenerations)

    completedReplicates = 0
    while completedReplicates < numReplicates:
        thisBatchSize = min(batchSize, numReplicates - completedReplicates)
        batchSims = getSimulatedTrajectories(thisInitScenario, thisPopScenario, thisBatchSize, numGenerations, numGeneticTypes, rho, commonMu)

        # first locus
        margOne = batchSims[:,:,0] + batchSims[:,:,1]
        sumOne += numpy.sum(margOne, axis=0)
        sumOneSq += numpy.sum(margOne**2, axis=0)
 
        hetOne = 2 * margOne * (1 - margOne)
        sumHetOne += numpy.sum(hetOne, axis=0)
        sumHetOneSq += numpy.sum(hetOne**2, axis=0)
 
        # second locus
        margTwo = batchSims[:,:,0] + batchSims[:,:,2]
        sumTwo += numpy.sum(margTwo, axis=0)
        sumTwoSq += numpy.sum(margTwo**2, axis=0)
 
        hetTwo = 2 * margTwo * (1 - margTwo)
        sumHetTwo += numpy.sum(hetTwo, axis=0)
        sumHetTwoSq += numpy.sum(hetTwo**2, axis=0)

        # LD, vectorized across all generations in batch
        thisNumRep, thisNumGen, thisNumTypes = batchSims.shape
        flatFreqs = batchSims.transpose(1, 0, 2).reshape(thisNumRep * thisNumGen, thisNumTypes)
        flatD = LD(flatFreqs)[:, 0]
        batchD = flatD.reshape(thisNumGen, thisNumRep).T
 
        sumD += numpy.sum(batchD, axis=0)
        sumD2 += numpy.sum(batchD**2, axis=0)
        sumD4 += numpy.sum(batchD**4, axis=0)

        completedReplicates += thisBatchSize
        percentDone = 100*(completedReplicates/numReplicates)
        print(f'-- {thisPopScenario} -- {thisInitScenario} -- {percentDone:.1f}% complete --')

    n = numReplicates
 
    meanOne = sumOne / n
    sdOne = numpy.sqrt(numpy.clip(sumOneSq/n - meanOne**2, 0, None))
 
    meanHetOne = sumHetOne / n
    sdHetOne = numpy.sqrt(numpy.clip(sumHetOneSq/n - meanHetOne**2, 0, None))
 
    meanTwo = sumTwo / n
    sdTwo = numpy.sqrt(numpy.clip(sumTwoSq/n - meanTwo**2, 0, None))
 
    meanHetTwo = sumHetTwo / n
    sdHetTwo = numpy.sqrt(numpy.clip(sumHetTwoSq/n - meanHetTwo**2, 0, None))
 
    meanLD = sumD / n
    sdLD = numpy.sqrt(numpy.clip(sumD2/n - meanLD**2, 0, None))
 
    meanLDSQ = sumD2 / n
    sdLDSQ = numpy.sqrt(numpy.clip(sumD4/n - meanLDSQ**2, 0, None))

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
    
    # batch to print progress updates and prevent memory issue
    numBatches = 25
    batchSize = max(1, numReplicates//numBatches)

    sumHigherOrder = {(dA, dB): numpy.zeros(numGenerations) for dA in numpy.arange(0, order+1) for dB in numpy.arange(0, order+1)}

    completedReplicates = 0
    while completedReplicates < numReplicates:
        thisBatchSize = min(batchSize, numReplicates - completedReplicates)
        batchSims = getSimulatedTrajectories(thisInitScenario, thisPopScenario, thisBatchSize, numGenerations, numGeneticTypes, rho, commonMu)

        for dA in numpy.arange(0, order+1):
            for dB in numpy.arange(0, order+1):
                batchVals = sum(multinomial([nAB, dA - nAB, dB - nAB, order + nAB - dA - dB]) * batchSims[:,:,0]**(nAB) * batchSims[:,:,1]**(dA - nAB) *
                                batchSims[:,:,2]**(dB - nAB) * batchSims[:,:,3]**(order + nAB - dA - dB) for nAB in range(max(0,dA + dB - order), min(dA,dB) + 1))
                sumHigherOrder[(dA, dB)] += numpy.sum(batchVals, axis=0)

        completedReplicates += thisBatchSize
        percentDone = 100*(completedReplicates/numReplicates)
        print(f'-- {thisPopScenario} -- {thisInitScenario} -- {percentDone:.1f}% complete --')

    simulatedHigherOrderTrajectories = {pair : sumVal/numReplicates for pair, sumVal in sumHigherOrder.items()}
    
    filename = WD + f'/sim.{thisInitScenario}.{thisPopScenario}.u={commonMu}.r={rho}.higherOrder={order}.pkl.bz2'
    
    ofs = bz2.open (filename, 'wb')
    pickle.dump (simulatedHigherOrderTrajectories, ofs)
    ofs.close()