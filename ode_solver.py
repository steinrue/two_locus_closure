# %%
import numpy
from scipy.integrate import odeint
import HigherOrderFunctions as higher
import variables

# 2 locus 2 allele mutation
# [locus, mutMatrix] 

def getMutMat(commonMu):

    allMu = [
        commonMu * numpy.array([[float("nan"), 1],
                                [1,           float("nan")]]),
        commonMu * numpy.array([[float("nan"), 1],
                                [1,           float("nan")]]),
    ]

    return allMu


# %%
def getAllStates(initialStates, order, perGenReco, commonMu):

    allMu = getMutMat(commonMu)

    statesToProcess = []
    statesToProcess += initialStates
    statesToProcess += higher.getlistG(order)
    statesToProcess = list(set(statesToProcess))


    driftTransitions = {}
    recoTransitions = {}
    mutTransitions = {}

    while (True):

        # where can we go from this state?
        srcState = statesToProcess.pop(0)


        # DRIFT
        thisDriftTransitions = {}

        # diagonal
        diagRate = - 0.5 * ( (srcState[0]*(srcState[0]-1) + srcState[1]*(srcState[1]-1) + srcState[2]*(srcState[2]-1)
                            + 2*srcState[0]*srcState[1] + 2*srcState[0]*srcState[2] + 2*srcState[1]*srcState[2]) )
        if (not numpy.isclose(diagRate, 0)):
            thisDriftTransitions[srcState] = diagRate

        # decrease a (coal in a)
        if (srcState[0] > 1):
            preDst = list(srcState)
            preDst[0] -= 1
            dstState = tuple(preDst) 

            thisDriftTransitions[dstState] = 0.5 * srcState[0]*(srcState[0]-1)

        # decrease b (coal in b or coal a with b)
        if (srcState[1] > 1 or ((srcState[0] >= 1) and (srcState[1] >= 1))):
            preDst = list(srcState)
            preDst[1] -= 1
            dstState = tuple(preDst)

            thisDriftTransitions[dstState] =  0.5 * (srcState[1]*(srcState[1]-1) + 2*srcState[0]*srcState[1])

        # decrease c (coal in c or coal a with c)
        if (srcState[2] > 1 or ((srcState[0] >= 1) and (srcState[2] >= 1))):
            preDst = list(srcState)
            preDst[2] -= 1
            dstState = tuple(preDst) 

            thisDriftTransitions[dstState] = 0.5 * (srcState[2]*(srcState[2]-1) + 2*srcState[0]*srcState[2])

        # coal b+c -> a
        if (srcState[1] >= 1 and srcState[2] >=1):
            preDst = list(srcState)
            preDst[0] += 1
            preDst[1] -= 1
            preDst[2] -= 1
            dstState = tuple(preDst) 

            thisDriftTransitions[dstState] = 0.5 * (2*srcState[1]*srcState[2])

        # did we add any?
        if (len(thisDriftTransitions) > 0):
            driftTransitions[srcState] = thisDriftTransitions


        # RECOMBINATION
        thisRecoTransitions = {}

        # diagonal
        diagRate = - perGenReco*srcState[0]

        if (not numpy.isclose(diagRate, 0)):
            thisRecoTransitions[srcState] = diagRate

        # break a -> b+c
        if (srcState[0] >= 1):
            preDst = list(srcState)
            preDst[0] -= 1
            preDst[1] += 1
            preDst[2] += 1
            dstState = tuple(preDst) 

            thisRecoTransitions[dstState] = perGenReco*srcState[0]

        # did we add any recombination transitions?
        if(len(thisRecoTransitions) > 1):
            recoTransitions[srcState] = thisRecoTransitions


        # MUTATION
        thisMutTransitions = {}

        # diagonal
        diagRate = - (allMu[0][0,1] + allMu[0][1,0]) * (srcState[0] + srcState[1]) - (allMu[1][0,1] + allMu[1][1,0]) * (srcState[0] + srcState[2])

        if (not numpy.isclose(diagRate, 0)):
            thisMutTransitions[srcState] = diagRate

        # -a +b
        if (srcState[0] >= 1):
            preDst = list(srcState)
            preDst[0] -= 1
            preDst[1] += 1
            dstState = tuple(preDst) 

            thisMutTransitions[dstState] = srcState[0] * allMu[1][1,0]

        # -a +c
        if (srcState[0] >= 1):
            preDst = list(srcState)
            preDst[0] -= 1
            preDst[2] += 1
            dstState = tuple(preDst) 

            thisMutTransitions[dstState] = srcState[0] * allMu[0][1,0]

        # -b
        if (srcState[1] >= 1):
            preDst = list(srcState)
            preDst[1] -= 1
            dstState = tuple(preDst) 

            thisMutTransitions[dstState] = srcState[1] * allMu[0][1,0]

        # -c
        if (srcState[2] >= 1):
            preDst = list(srcState)
            preDst[2] -= 1
            dstState = tuple(preDst) 

            thisMutTransitions[dstState] = srcState[2] * allMu[1][1,0]

        # did we add any mutation transitions
        if (len(thisMutTransitions) > 0):
            mutTransitions[srcState] = thisMutTransitions


        # LOOP
        # are there any new states? in any transitions?
        # we have to use the potential dicts
        potentialNewToProcess = (list(thisDriftTransitions.keys())
                                    + list(thisRecoTransitions.keys())
                                    + list(thisMutTransitions.keys()))

        for thisDst in potentialNewToProcess:
            inDrift = (thisDst in driftTransitions)
            inReco = (thisDst in recoTransitions)
            inMut = (thisDst in mutTransitions)
            inAny = inDrift or inReco or inMut
            # if it is in any of them, we have processed it completely
            if (not inAny):
                statesToProcess.append (thisDst)
        
        if (len(statesToProcess) <= 0):
            return driftTransitions, recoTransitions, mutTransitions


# %%
# collect all states from all transitions
# nobody has (0,0,0), so add it manually

def getIdxStatePairs(driftTransitions, recoTransitions, mutTransitions):

    idxToState = [(0,0,0)] + list(set(
        list(driftTransitions)
        + list(recoTransitions)
        + list(mutTransitions)
    ))

    stateToIdx = {}
    for (idx, state) in enumerate(idxToState):
        stateToIdx[state] = idx
    
    return idxToState, stateToIdx



# %%
# build the matrices

def getMatrixQ(driftTransitions, recoTransitions, mutTransitions):

    idxToState, stateToIdx = getIdxStatePairs(driftTransitions, recoTransitions, mutTransitions)
    numStates = len(idxToState)

    Qs = {
        "drift" : numpy.zeros ((numStates, numStates), dtype=float),
        "reco" : numpy.zeros ((numStates, numStates), dtype=float),
        "mut" : numpy.zeros ((numStates, numStates), dtype=float),
    }

    # to iterate more easily
    allTransitions ={
        "drift" : driftTransitions,
        "reco" : recoTransitions,
        "mut" : mutTransitions,
    }

    # go through generator types
    for (genType, thisTransitions) in allTransitions.items():
        thisQ = Qs[genType]
        # fill in the values
        for (srcState, destinations) in thisTransitions.items():
            for (dstState, rate) in destinations.items():
                thisQ[stateToIdx[srcState],stateToIdx[dstState]]= rate
        assert (not numpy.any(numpy.isnan(thisQ)))

    return Qs

# %%

def getStationaryMoments(thisPopScenario, initialStates, order, perGenReco, mu):
    driftTransitions, recoTransitions, mutTransitions = getAllStates(initialStates, order, perGenReco, mu)
    idxToState, stateToIdx = getIdxStatePairs(driftTransitions, recoTransitions, mutTransitions)
    Qs = getMatrixQ(driftTransitions, recoTransitions, mutTransitions)

    if thisPopScenario == 'constant' or thisPopScenario == 'bottleneck':
        Ne0 = variables.popScenarios[thisPopScenario][0]
        fullQ = ((1/(2 * Ne0))*Qs["drift"] + Qs["reco"] + Qs["mut"])
    elif thisPopScenario == 'expGrowth':
        Ne0 = variables.popScenarios[thisPopScenario][0]
        fullQ = ((1/(2 * Ne0))*Qs["drift"] + Qs["reco"] + Qs["mut"])
    else:
        raise ValueError("Unknown demography type")
    # try to get stationarity by solving a linear system
    # the system is: Q * x + b
    # where Q is matrix without row/col for (0,0,0)
    # and b is 1 * col for (0,0,0)
    zeroIdx = stateToIdx[(0,0,0)]
    # delete row and col that correpsond to (0,0,0)
    reducedQ = numpy.delete (numpy.delete (fullQ, zeroIdx, axis=0), zeroIdx, axis=1)
    assert(numpy.array(idxToState)[numpy.all(numpy.isclose(fullQ, 0), axis=0)].size == 0)
    assert((0,0,0) in numpy.array(idxToState)[numpy.all(numpy.isclose(fullQ, 0), axis=1)])
    assert (reducedQ.shape[0] == reducedQ.shape[1])
    assert (reducedQ.shape[0] == numpy.linalg.matrix_rank (reducedQ))
    # remeber the column of Q that corresponds to (0,0,0)
    inhom = numpy.delete (fullQ[:,zeroIdx], zeroIdx)
    # and do some solving
    stationaryG = numpy.linalg.solve (reducedQ, -inhom)

    return stationaryG, stateToIdx



# %%
def solveODE(thisInitScenario, thisPopScenario, initialStates, order, time, perGenReco, mu):
    
    driftTransitions, recoTransitions, mutTransitions = getAllStates(initialStates, order, perGenReco, mu)
    idxToState, stateToIdx = getIdxStatePairs(driftTransitions, recoTransitions, mutTransitions)
    Qs = getMatrixQ(driftTransitions, recoTransitions, mutTransitions)

    # set initial frequencies
    if thisInitScenario != 'stat':
        x0 = variables.allScenarios[thisInitScenario]['x0']
        z0 = [x0[0], x0[0]+x0[1], x0[0]+x0[2]]

    # initialize G0
    if thisInitScenario == 'stat':
        assert(mu != 0)
        tempG0, tempStateToIdx = getStationaryMoments(thisPopScenario, initialStates, order, perGenReco, mu)
        G0 = list(tempG0)
        G0.insert(0,1)
    else:
        G0 = []
        for exp in idxToState:
            G0.append(z0[0]**exp[0] * z0[1]**exp[1] * z0[2]**exp[2])

    if thisPopScenario == 'constant':
        def ode(G,t):
            Ne = variables.popScenarios[thisPopScenario][0]
            dGdt = ((1/(2 * Ne))*Qs["drift"] + Qs["reco"] + Qs["mut"])@G
            return dGdt
    elif thisPopScenario == 'bottleneck':
        def ode(G,t):
            # Clamp and interpolate Ne for time t
            if t < 0:
                Ne = variables.popScenarios[thisPopScenario][0]
            elif t >= len(variables.popScenarios[thisPopScenario]):
                Ne = variables.popScenarios[thisPopScenario][-1]
            else:
                Ne = numpy.interp(t, numpy.arange(len(variables.popScenarios[thisPopScenario])), variables.popScenarios[thisPopScenario])
            dGdt = ((1/(2 * Ne)) * Qs["drift"] + Qs["reco"] + Qs["mut"])@G
            return dGdt
    elif thisPopScenario == 'expGrowth':
        def ode(G,t):
            # Clamp and interpolate Ne for time t
            if t < 0:
                Ne = variables.popScenarios[thisPopScenario][0]
            elif t >= len(variables.popScenarios[thisPopScenario]):
                Ne = variables.popScenarios[thisPopScenario][-1]
            else:
                Ne = numpy.interp(t, numpy.arange(len(variables.popScenarios[thisPopScenario])), variables.popScenarios[thisPopScenario])
            dGdt = ((1/(2 * Ne)) * Qs["drift"] + Qs["reco"] + + Qs["mut"])@G
            return dGdt

    time = numpy.linspace(0, 2000, 200)
    G=numpy.transpose(odeint(ode,G0,time, hmax=1))

    return G, stateToIdx
