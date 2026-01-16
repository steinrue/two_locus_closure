# Import statements
import numpy as np
import msprime
import scipy
import bz2
import pickle
import variables

# random seed
metaRNG = np.random.default_rng(8213)

def getMsprimeDemographies():
    # build different demographies for msprime
    # SINCE WE TAKE A HAPLOID SAMPLE IN MSPRIME, THE POPULATION SIZES ALSO HAVE TO BE HAPLOID
    haploidFactor = 2
    # constant
    constDemography = msprime.Demography()
    constDemography.add_population (name='pop0', initial_size=haploidFactor*variables.constantSize)
    # bottleneck (keep in mind that time here increases backwards from the present (t=0))
    bottleneckDemography = msprime.Demography()
    bottleneckDemography.add_population (name='pop0', initial_size=haploidFactor*variables.bottleneckFull)
    bottleneckDemography.add_population_parameters_change (population='pop0', time=variables.bottleneckTimeFull, initial_size=haploidFactor*variables.bottleneckShrink)
    bottleneckDemography.add_population_parameters_change (population='pop0', time=variables.bottleneckTimeFull+variables.bottleneckTimeShrink, initial_size=haploidFactor*variables.bottleneckFull)
    # exponential growth
    expGrowthDemography = msprime.Demography()
    # the rate is set such that it goes in 100 generations from 500 to 40000 (80-fold increase)
    expGrowthDemography.add_population (name='pop0', initial_size=haploidFactor*variables.expFinalSize, growth_rate=(variables.expGrowthRate))
    expGrowthDemography.add_population_parameters_change (population='pop0', time=variables.expGrowthTime, initial_size=haploidFactor*variables.expGrowthStart, growth_rate=0)
    # collect in dictionary for convenient access
    demographies = {
        'constant' : constDemography,
        'bottleneck' : bottleneckDemography,
        'expGrowth' : expGrowthDemography,
    }
    return demographies

def getMutRecoString (thisMutRate, thisRecoRate):
    return f"m_{thisMutRate:.5e}_r{thisRecoRate:.5e}"

def msprimePairwiseSimulations (haploidSampleSize, recoRates, mutRates, demographies, numPairwiseReplicates):

    # for convenience
    n = haploidSampleSize

    # iterate over demographies
    jSFSs = {}
    jSFSs_relErr = {}
    for (thisDemoLab, thisDemography) in demographies.items():
        print (f"-- {thisDemoLab}")

        jSFSs[thisDemoLab] = {}
        jSFSs_relErr[thisDemoLab] = {}

        for thisMutRate in mutRates:
            for thisRecoRate in recoRates:
                print (f"---- {getMutRecoString(thisMutRate, thisRecoRate)}")

                # storage for results
                # simJointSFS = np.zeros ((n+1,n+1), dtype=int)
                replicateSFSs = np.zeros((numPairwiseReplicates, n+1, n+1))

                # simulate the replicates
                for r in np.arange(numPairwiseReplicates):
                    # print (f'===== {r}')

                    # simulate ancestry using msprime
                    # we set ploidy 1 for haploid samples
                    ts = msprime.sim_ancestry(
                        samples=n,
                        ploidy=1,
                        recombination_rate=thisRecoRate,
                        sequence_length=2,
                        demography=thisDemography,
                        random_seed=metaRNG.integers(99999999),
                    )

                    # and mutations on the ancestry
                    mutTs = msprime.sim_mutations(
                        ts,
                        rate=thisMutRate,
                        model=msprime.BinaryMutationModel(state_independent=False),
                        random_seed=metaRNG.integers(99999999)
                    )

                    # record the allele counts for this replicate
                    simGeno = mutTs.genotype_matrix()
                    simPositions = mutTs.sites_position.astype(int)
                    assert (len(simPositions) <= 2)
                    assert (simGeno.shape[1] == n)

                    # first position
                    d1 = 0
                    if (0 in simPositions):
                        p1Idx =np.where(simPositions == 0)[0][0]
                        d1 = simGeno[p1Idx].sum()

                    # second position
                    d2 = 0
                    if (1 in simPositions):
                        p2Idx = np.where(simPositions == 1)[0][0]
                        d2 = simGeno[p2Idx].sum()

                    # tally this simulation in correct bin
                    # simJointSFS[min(d1,n-d1), min(d2,n-d2)] += 1
                    replicateSFSs[r, min(d1, n-d1), min(d2, n-d2)] += 1

                # normalize SFS
                # normJointSFS = simJointSFS / simJointSFS.sum()
                replicateSFSs /= np.sum(replicateSFSs, axis=(1,2), keepdims=True)

                # # symmetrize over order of two loci
                # normJointSFS = (normJointSFS + normJointSFS.transpose()) / 2

                # # symmetrize over MAFs
                # vFlip = np.flip (normJointSFS, axis=1)
                # hFlip = np.flip (normJointSFS, axis=0)
                # ddFlip = np.flip (np.flip(normJointSFS, axis=0), axis=1)
                # normJointSFS = (normJointSFS + vFlip + hFlip + ddFlip) / 4

                for r in np.arange(numPairwiseReplicates):
                    # Symmetrize over order of two loci
                    mat = replicateSFSs[r]
                    mat = (mat + mat.transpose()) / 2
                    # Symmetrize over MAFs
                    vFlip = np.flip(mat, axis=1)
                    hFlip = np.flip(mat, axis=0)
                    ddFlip = np.flip(hFlip, axis=1)
                    replicateSFSs[r] = (mat + vFlip + hFlip + ddFlip) / 4
                
                # Compute mean and relative error
                meanSFS = np.mean(replicateSFSs, axis=0)
                sdSFS = np.std(replicateSFSs, axis=0, ddof=1)
                relErrSFS = np.divide(sdSFS/np.sqrt(numPairwiseReplicates), np.abs(meanSFS) + 1e-12)

                # and record the SFS
                thisMutRecString = getMutRecoString(thisMutRate, thisRecoRate)
                jSFSs[thisDemoLab][thisMutRecString] = meanSFS
                jSFSs_relErr[thisDemoLab][thisMutRecString] = relErrSFS

    # should be all in the dictionary
    return jSFSs, jSFSs_relErr

def storeJSFSs (pairwiseJSFSs, pairwiseRelErr, genomicJSFSs, genomicRelErr, pickleBz2Filename):

    pickleData = {
        "pairwiseJSFSs" : pairwiseJSFSs,
        "pairwiseRelErr" : pairwiseRelErr,
        "genomicJSFSs" : genomicJSFSs,
        "genomicRelErr": genomicRelErr
    }
    pickleFile = bz2.open(pickleBz2Filename, 'wb')
    pickle.dump(pickleData, pickleFile)
    pickleFile.close()

def getRecoIdx (thisDist, recoLBs, recoUBs):
    # +1 deals correctly with boundary cases
    lbIdx = np.searchsorted(recoLBs, thisDist+1)
    ubIdx = np.searchsorted(recoUBs, thisDist+1)
    assert (ubIdx <= lbIdx)
    if (ubIdx + 1 == lbIdx):
        # only valid one
        return ubIdx
    else:
        return None

def runGenomeSim(thisDemography, haploidSampleSize, genomicRecoRate, thisMutRate, sequenceLength, recoLBs, recoMids, recoUBs):
    n = haploidSampleSize

    # simulate ancestry
    # we set ploidy 1 for haplod samples
    ts = msprime.sim_ancestry(
        samples=haploidSampleSize,
        ploidy=1,
        recombination_rate=genomicRecoRate,
        sequence_length=sequenceLength,
        demography=thisDemography,
        random_seed=metaRNG.integers(99999999),
    )

    # and then superimpose mutations
    mutTs = msprime.sim_mutations(
        ts,
        rate=thisMutRate,
        model=msprime.BinaryMutationModel(state_independent=False),
        random_seed=metaRNG.integers(99999999)
    )

    # prepare empty count matrices
    countMatrix = np.zeros((len(recoMids), haploidSampleSize+1, haploidSampleSize+1))
    # just count regular SFS as well
    sfs = np.zeros(haploidSampleSize+1, dtype=int)

    # extract the right pairs
    simGeno = mutTs.genotype_matrix()
    simPositions = mutTs.sites_position.astype(int)
    # just making sure
    assert (simGeno.shape == (len(simPositions), haploidSampleSize))
    assert (np.diff(simPositions).min() >= 1)

    # only keep biallelic positions
    biallelicMask = (simGeno.max(axis=1) <= 1)
    assert (np.all(biallelicMask))
    # and segregating
    simAlleleCounts = simGeno.sum(axis=1)
    segregatingMask = (simAlleleCounts > 0) & (simAlleleCounts < haploidSampleSize)
    realPositions = simPositions[biallelicMask & segregatingMask]
    realAlleleCounts = simAlleleCounts[biallelicMask & segregatingMask]
    assert (len(realAlleleCounts) == len(realPositions))

    # now enumerate all relevant pairs (both segregating)
    for i in np.arange(len(realPositions)):
        posA = realPositions[i]
        dA = realAlleleCounts[i]
        # print (posA)
        sfs[min(dA,n-dA)] += 1

        # iterate B position, but only up to maximal bound
        minJ = max(i+1, np.searchsorted(realPositions, posA+recoLBs.min()))
        maxJ = np.searchsorted(realPositions, posA+recoUBs.max())
        # and only positive, because we would double-count otherwise
        # for j in numpy.arange(i+1,maxJ):
        for j in np.arange(minJ,maxJ):
            posB = realPositions[j]
            dB = realAlleleCounts[j]
            # print (posB)

            # what's the distance between A and B?
            distAB = posB - posA
            recoIdx = getRecoIdx(distAB, recoLBs, recoUBs)
            if (recoIdx is None):
                # no bin for this distance, try again
                continue

            # count this pair in the right place
            # perhaps only keep track of the MAF (let's actaully try the real counts), and do MAF stuff later
            # and also only ordered config
            countMatrix[recoIdx,min(dA,n-dA),min(dB,n-dB)] += 1
            # thisCounts[dA,dB] += 1

    # print (countMatrix[0])
    # print (countMatrix[0].sum())

    # get correct numbers for only one segregating or none segregating
    potentialTotalPairs = np.zeros(len(recoMids))
    for (thisRecoIdx, thisRecoMid) in enumerate(recoMids):
        thisIntervalSpan = recoUBs[thisRecoIdx] - recoLBs[thisRecoIdx]
        # count total number of potential pairs in this recombination interval
        lastFullUpperIndex = sequenceLength - recoUBs[thisRecoIdx]
        # these have all their partners
        numUpperPairs = lastFullUpperIndex * thisIntervalSpan
        # and then we have a triangle for the remainder
        numUpperPairs += int(thisIntervalSpan * (thisIntervalSpan-1) / 2)
        # since things are symmetric, the lower pairs look the same
        potentialTotalPairs[thisRecoIdx] = numUpperPairs

    # print (potentialTotalPairs[0])

    # now each variant creates a bunch of potential pairs
    potentialLowerPairs = np.zeros((len(recoMids), haploidSampleSize+1), dtype=int)
    potentialUpperPairs = np.zeros((len(recoMids), haploidSampleSize+1), dtype=int)
    # go through variants and see what they do
    for i in np.arange(len(realPositions)):
        posA = realPositions[i]
        dA = realAlleleCounts[i]

        # see about the number of potential pairs this variant creates in each reco bin
        for (thisRecoIdx, thisRecoMid) in enumerate(recoMids):
            # print (posA, recoLBs[thisRecoIdx], recoUBs[thisRecoIdx], sequenceLength)
            possbileLowerPartners = max(0,posA-recoLBs[thisRecoIdx]) - max(0,posA-recoUBs[thisRecoIdx])
            possbileUpperPartners = min(sequenceLength,posA+recoUBs[thisRecoIdx]) - min(sequenceLength,posA+recoLBs[thisRecoIdx])
            # print (possbileLowerPartners, possbileUpperPartners)
            # 0.5 because it could be lower or upper?
            potentialLowerPairs[thisRecoIdx,min(dA,n-dA)] += possbileLowerPartners
            potentialUpperPairs[thisRecoIdx,min(dA,n-dA)] += possbileUpperPartners

    # I think we might be able to put a full count matrix together now
    fullCountMatrix = countMatrix.copy()
    for (thisRecoIdx, thisRecoMid) in enumerate(recoMids):
        # now the potential lower and upper pairs are on the edges, but we have to take away stuff that ended up in the interior
        # but a certain segregating site was used as lower site sometimes, but also as upper site at other times
        # so we make this a bit approximate here
        rowSum = countMatrix[thisRecoIdx].sum(axis=0)
        colSum = countMatrix[thisRecoIdx].sum(axis=1)
        meanNumPairs = (rowSum + colSum) / 2

        # half of them upper and half of them lower
        fullCountMatrix[thisRecoIdx,0,:] += (potentialLowerPairs[thisRecoIdx] - meanNumPairs)
        fullCountMatrix[thisRecoIdx,:,0] += (potentialUpperPairs[thisRecoIdx] - meanNumPairs)

        # these are the potential pairs that each segregating site creates
        potentialSegPairs = potentialLowerPairs[thisRecoIdx].sum() + potentialUpperPairs[thisRecoIdx].sum()
        # so we have to take those away from the corner (0,0)
        # WARNING: this has to e after previous code, because we change v as we go (now that is is renamed, maybe not)
        fullCountMatrix[thisRecoIdx,0,0] = potentialTotalPairs[thisRecoIdx] - fullCountMatrix[thisRecoIdx].sum()

    # now we account for all the symmetries
    symmetricMatrix = fullCountMatrix.copy()
    for (thisRecoIdx, thisRecoMid) in enumerate(recoMids):

        # firstly, we don't know who upper and lower partner is
        symmetricMatrix[thisRecoIdx] = (symmetricMatrix[thisRecoIdx] + symmetricMatrix[thisRecoIdx].transpose()) / 2

        # we also have to put mass computed with MAFs into the full matrix
        # so flip some things
        hFlip = np.flip(symmetricMatrix[thisRecoIdx], axis=0)
        vFlip = np.flip(symmetricMatrix[thisRecoIdx], axis=1)
        doubleFlip = np.flip(np.flip(symmetricMatrix[thisRecoIdx], axis=1), axis=0)

        # and take the mean
        symmetricMatrix[thisRecoIdx] = (symmetricMatrix[thisRecoIdx] + hFlip + vFlip + doubleFlip) / 4

        # and normalize
        symmetricMatrix[thisRecoIdx] /= symmetricMatrix[thisRecoIdx].sum()
    
    # record total observed successes per bin
    numSuccessPerBin = np.sum(fullCountMatrix, axis=(1,2))
    # see about empirical sfs
    modSFS = sfs.copy()
    modSFS[0] = sequenceLength - modSFS.sum()
    modSFS = modSFS / modSFS.sum()
    # symmetrize MAF
    modSFS = (modSFS + np.flip(modSFS)) / 2

    return symmetricMatrix, modSFS, potentialTotalPairs, numSuccessPerBin


def msprimeGenomicSimulations (haploidSampleSize, mutRates, sequenceLength, genomicRecoRate, recoLBs, recoMids, recoUBs, demographies, numGenomicReplicates):
    # check if bounds look all good
    assert (len(recoMids) == len(recoLBs))
    assert (len(recoMids) == len(recoUBs))
    assert (np.all(recoLBs < recoMids))
    assert (np.all(recoMids < recoUBs))
    # at most just directly adjacent
    assert (np.all (recoUBs[:-1] <= recoLBs[1:]))
    assert (recoLBs[0] >= 0)

    # for convenience
    n = haploidSampleSize

    # iterate over demographies
    jSFSs = {}
    jSFSs_relErr = {}
    for (thisDemoLab, thisDemography) in demographies.items():
        print (f"-- {thisDemoLab}")

        jSFSs[thisDemoLab] = {}
        jSFSs_relErr[thisDemoLab] = {}

        for thisMutRate in mutRates:

            # for all the replicates
            replicateMatrix = None
            replicateSFS = None
            replicateMatrices = []
            replicateSFSs = []
            replicatePotentialPairs = []

            # go through replicates
            for r in np.arange(numGenomicReplicates):
                symmetricMatrix, modSFS, potentialTotalPairs, _ = runGenomeSim(thisDemography, n, genomicRecoRate, thisMutRate, sequenceLength, recoLBs, recoMids, recoUBs)
                replicateMatrices.append(symmetricMatrix)
                replicateSFSs.append(modSFS)
                replicatePotentialPairs.append(potentialTotalPairs)

                # add it to average across replicates
                if (replicateMatrix is None):
                    replicateMatrix = symmetricMatrix
                else:
                    replicateMatrix += symmetricMatrix

                if (replicateSFS is None):
                    replicateSFS = modSFS
                else:
                    replicateSFS += modSFS

            # all replicates done, average over replicates
            assert (replicateMatrix is not None)
            replicateMatrix /= numGenomicReplicates
            assert (replicateSFS is not None)
            replicateSFS /= numGenomicReplicates

            # compute relative error
            replicateMatrices = np.stack(replicateMatrices)
            meanSFS = replicateMatrix
            sdSFS = np.std(replicateMatrices, axis=0, ddof=1)
            meanPotentialPairs = np.mean(np.stack(replicatePotentialPairs), axis=0)
            relErrSFS = np.zeros_like(sdSFS)
            for thisRecoIdx in range(len(recoMids)):
                N_eff = numGenomicReplicates*meanPotentialPairs[thisRecoIdx]
                relErrSFS[thisRecoIdx] = np.divide(sdSFS[thisRecoIdx]/np.sqrt(N_eff), np.abs(meanSFS[thisRecoIdx]) + 1e-12)

            # and store the right things with the right labels
            for (thisRecoIdx, thisRecoMid) in enumerate(recoMids):
                thisMutRecString = getMutRecoString(thisMutRate, thisRecoMid * genomicRecoRate)
                jSFSs[thisDemoLab][thisMutRecString] = replicateMatrix[thisRecoIdx]
                jSFSs_relErr[thisDemoLab][thisMutRecString] = relErrSFS[thisRecoIdx]

    # should be all stored in the right place
    return jSFSs, jSFSs_relErr

def runSimulations():

    # parameters for joint SFS to estimate
    haploidSampleSize = variables.ploidy
    mutRates = variables.msprimeMuts

    print ("[CREATE_DEMOGRAPHIES]")
    demographies = getMsprimeDemographies()
    print ("[CREATE_DEMOGRAPHIES_DONE]")

    pairwiseJSFSs = None
    print ("[PAIRWISE_SIMULATIONS]")
    numPairwiseReplicates = variables.pairwiseReplicates
    # recoRates = [1.25e-4, 5e-4]
    recoRates = variables.pairwiseReco
    pairwiseJSFSs, pairwiseRelErr = msprimePairwiseSimulations(haploidSampleSize, recoRates, mutRates, demographies, numPairwiseReplicates)
    print ("[PAIRWISE_SIMULATIONS_DONE]")


    genomicJSFSs = None
    print ("[GENOMIC_SIMULATIONS]")
    # mutRates has to be the same as for the other simulation, otherwise the two-allele recurrent model does not make sense
    numGenomicReplicates = variables.genomeReplicates
    sequenceLength = variables.seqLength
    genomicRecoRate = variables.genomicReco
    # numBins = 10
    # preBounds = numpy.geomspace(0.5, int(1e6), (2*numBins + 1)).astype(int)
    # preBounds = numpy.linspace(0, int(1e6), (2*numBins + 1)).astype(int)
    # recoLBs = preBounds[:-1:2]
    # recoMids = preBounds[1::2]
    # recoMids = numpy.linspace(int(1e5), int(3e5), 3).astype(int)
    recoLBs = variables.recoMids - variables.pm
    recoUBs = variables.recoMids + variables.pm
    genomicJSFSs, genomicRelErr = msprimeGenomicSimulations(haploidSampleSize, mutRates, sequenceLength, genomicRecoRate, recoLBs, variables.recoMids, recoUBs, demographies, numGenomicReplicates)
    print ("[GENOMIC_SIMULATIONS_DONE]")


    print ("[SAVE_PICKLE]")
    # pickleBz2Filename = "msprime_results.pickle.bz2"
    pickleBz2Filename = "pickles/msprime_simulations.pickle.bz2"
    storeJSFSs(pairwiseJSFSs, pairwiseRelErr, genomicJSFSs, genomicRelErr, pickleBz2Filename)
    print ("[SAVE_PICKLE_DONE]")

def main():
    runSimulations()

if __name__ == "__main__":
    main()