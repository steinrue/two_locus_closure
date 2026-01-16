# %%
import numpy
import matplotlib.pyplot as plt
import pickle
import bz2
import variables

metaRNG = numpy.random.default_rng (4711)

EPSILON = 1e-12

# %%
# encoding: (11, 10, 01, 00)

# marginals
# 0 -- 1* = 11 + 10
# 1 -- 0* = 01 + 00
# 2 -- *1 = 11 + 01
# 3 -- *0 = 10 + 00

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

# %%
# encoding: (11, 10, 01, 00)

# marginals
# 0 -- 1* = 11 + 10
# 1 -- 0* = 01 + 00
# 2 -- *1 = 11 + 01
# 3 -- *0 = 10 + 00

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

# %%
def LD (p):
    return p - getFreeCombinations (getMarginals (p))

# %%
# encoding: (11, 10, 01, 00)
def mutationMatrix (mu):
    
    Q = numpy.zeros((4,4))
    # 11 -> (01, 10) at rate (mu[0][1], mu[1][1])
    Q[0,2] = mu[0][1]
    Q[0,1] = mu[1][1]
    # 10 -> (00, 11) at rate (mu[0][1], mu[1][0])
    Q[1,3] = mu[0][1]
    Q[1,0] = mu[1][0]
    # 01 -> (11, 00) at rate (mu[0][0], mu[1][1])
    Q[2,0] = mu[0][0]
    Q[2,3] = mu[1][1]
    # 00 -> (10, 01) at rate (mu[0][0], mu[1][0])
    Q[3,1] = mu[0][0]
    Q[3,2] = mu[1][0]

    # set diagonal to - off-diagonals
    Q -= numpy.diag(numpy.sum(Q, axis=1))

    return Q

# %%
# either for loop over scenarios
# or just specfiy one

def getSimulatedTrajectories(thisInitScenario, thisPopScenario, numReplicates, numGenerations, numGeneticTypes, rho, commonMu):


    Ne = variables.popScenarios[thisPopScenario]
    initFreqs = variables.allScenarios[thisInitScenario]['x0']
    thisRNG = numpy.random.default_rng (metaRNG.integers (99999999))    
    
    

    # set up empty trajectories
    simulatedTrajectories = -1 * numpy.ones ((numReplicates, numGenerations, numGeneticTypes))

    # broadcast initial condition to all replicates
    simulatedTrajectories[:,0,:] = numpy.tile (initFreqs, (numReplicates, 1))
    
    # [locus, sourceAllele] 
    allMu = numpy.array([
        [commonMu, commonMu],
        [commonMu, commonMu],
    ])

    # need the mutation matrix
    mutationQ = mutationMatrix (allMu)

    # and step by step updates over generations
    for nextGen in numpy.arange (1,simulatedTrajectories.shape[1]):

        # get the frequncies before reco
        prevFrequencies = simulatedTrajectories[:,nextGen-1,:]

        # mutation
        # this could also be Q@p
        afterMutFrequencies = prevFrequencies + prevFrequencies@mutationQ

        # recombination
        afterRecoFrequencies = (1 - rho) * afterMutFrequencies + rho * getFreeCombinations (getMarginals (afterMutFrequencies))

        # genetic drift
        # nextFrequencies = 1/(2*Ne[nextGen-1]) * thisRNG.multinomial (2*Ne[nextGen-1], afterRecoFrequencies)
        intNe = int(round(Ne[nextGen]))
        nextFrequencies = 1/(2*intNe) * thisRNG.multinomial (2*intNe, afterRecoFrequencies)

        simulatedTrajectories[:,nextGen,:] = nextFrequencies

    # just to be sure, for now ok, maybe EPSILON in future
    assert (simulatedTrajectories.min() >= 0), simulatedTrajectories.min()
    assert (simulatedTrajectories.max() <= 1), simulatedTrajectories.max()
    assert (numpy.all (numpy.isclose (numpy.sum(simulatedTrajectories, axis=2), 1)))

    return simulatedTrajectories
