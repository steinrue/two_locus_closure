# %%
import numpy
from scipy.special import binom

# %%
#order = 5

# initializing the polynomial p^n(dA,dB) in the Ms
def getInputStates(dA,dB, order):
    inputStates = []
    for nAB in range(max(0,dA + dB - order) , min(dA,dB) + 1):
        inputStates.append((nAB, dA - nAB, dB - nAB, order + nAB - dA - dB, 1))
    return inputStates



# statesToProcess = inputStates
# outputStates = []

# %%
#converting the Ms to have M_{22} = 0
def getOutputStates(dA, dB, order):
    statesToProcess = getInputStates(dA,dB, order)
    outputStates = []

    while(len(statesToProcess) > 0):
        thisState = statesToProcess.pop(0)

        if thisState[len(thisState)-2] == 0:
            outputStates.append(thisState)
        else:
            np1 = numpy.sum (thisState[:-1])
            for thisCoordinate in range(len(thisState)-2):
                oneHigher = list(thisState)
                oneHigher[thisCoordinate] += 1
                oneHigher[len(thisState)-2] -= 1
                # oneHigher[len(thisState)-1] *= -1
                oneHigher[len(thisState)-1] *= - oneHigher[thisCoordinate] / (oneHigher[len(thisState)-2]+1)
                statesToProcess.append(tuple(oneHigher))
            oneLower = list(thisState)
            oneLower[len(thisState)-2] -= 1
            # oneLower[len(thisState)-1] *= 1
            oneLower[len(thisState)-1] *= np1 / (oneLower[len(thisState)-2]+1)
            statesToProcess.append(tuple(oneLower))

    return outputStates


# %%
# function to enumerate all tuples of certain length and certain sum
# from stack overflow
def getTuples(length, order):
    if length == 1:
        yield (order,)
        return

    for i in range(order + 1):
        for t in getTuples(length - 1, order - i):
            yield (i,) + t


# %%
# listing all possible exponent triples for Gs of order less than or equal to given order
def getlistG(order):
    listG = []
    for i in range(0, order + 1):
        listG.extend(list(getTuples(3,i)))

    #listG.append((0,0,0))
    return listG

# %%
#listing all possible exponent quadruples for Ms with M_{22} = 0 of order less than or equal to given order

def getlistM0(order):
    listM0 = []
    for thisG in getlistG(order):
        temp = list(thisG)
        temp.append(0)
        listM0.append(tuple(temp))
    
    return listM0


# %%
# useful conversions

def getidxToStateM0(order):
    return getlistM0(order)


def getstateToIdxM0(order):
    stateToIdxM0 = {}
    for (idx, state) in enumerate(getidxToStateM0(order)):
        stateToIdxM0[state] = idx
    
    return stateToIdxM0

# %%
#the coefficients of the Ms from our polynomial p^(n)(dA,dB), but with n_{22} = 0

def getCoeffsM0(dA,dB, order):
    outputStates = getOutputStates(dA,dB, order)
    listM0 = getlistM0(order)
    stateToIdxM0 = getstateToIdxM0(order)

    coefficientsM0 = numpy.full(len(listM0), 0 , dtype = float)
    for thisState in outputStates:
        coefficientsM0[stateToIdxM0[thisState[0:len(thisState)-1]]] += thisState[-1]
    return coefficientsM0

# %%
# multinomial function (copied from stack exchange)

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])

# %%
# Creating the Matrix


def getB(order):

    listM0 = getlistM0(order)
    stateToIdxM0 = getstateToIdxM0(order)

    # initializing
    B = numpy.full((len(listM0),len(listM0)), 0, dtype=float)

    #plugging in values
    for thisState in listM0:
        alpha = thisState[0]
        beta = thisState[1]
        gamma = thisState[2]
        for i in range(beta + 1):
            for j in range(gamma + 1):
                indexState = list(thisState)
                indexState[0] += i+j
                indexState[1] -= i
                indexState[2] -= j
                B[stateToIdxM0[thisState] , stateToIdxM0[tuple(indexState)]] = ( binom(beta, i)*binom(gamma,j) ) / multinomial([alpha + i + j, beta - i, gamma - j])
    
    return B

# %%

# filter B appropriately

def getRelevantStates(order, stateToIdx):
    relevantStates = []
    idxToStateM0 = getidxToStateM0(order)
    for thisState in idxToStateM0:
        tempState = list(thisState)
        tempState.pop(-1)
        relevantStates.append(stateToIdx[tuple(tempState)])
    
    return relevantStates



# %%
def jointProb(dA, dB, order, G, stateToIdx):
    coefficientsM0 = getCoeffsM0(dA,dB, order)
    filterIndeces = getRelevantStates(order, stateToIdx)
    # listM0 = getlistM0(order)
    # stateToIdxM0 = getstateToIdxM0(order)
    B = getB(order)
    G = G[filterIndeces,:]
    #coefficientsM0 = coefficientsM0[filterIndeces]
    return numpy.dot(coefficientsM0, numpy.linalg.inv(B)@G)

