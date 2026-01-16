
# General Moment Closure for Two-Locus Wright-Fisher Dynamics

This repository contains the code to run the analyses and create the figures for the manuscript:
>   Kundragami, R., Yetter, S., and Steinrücken, M. (2025) General moment closure for the neutral two-locus Wright-Fisher dynamics. https://doi.org/xyz

In this manuscript, we rewrite the coordinates used to describe the Wright-Fisher diffusion of allele frequencies for two loci. We rederive the generator of the process in these coordinates, and show that the moment equations for genetic drift, recurrent mutation, and recombination close in this new coordinate system. Accordingly, we are able to numerically integrate the system of ODEs arising from these moment equations. This code performs these integrals and other simulations, producing the figures from the manuscript to demostrate the accuracy and efficiency of this framework.

## Installation

Run this to clone and enter the directory with the code:
```bash
git clone https://github.com/steinrue/two_locus_closure.git
cd two_locus_closure
```
To run the scripts, the modules `numpy`, `scipy`, `seaborn`, `pickle`, `bz2`, `matplotlib`, `string` and `msprime` need to be available in python.

## Usage

The process of generating the figures for the main text has been streamlined in this pipeline. Perform the following to recreate Figures 2 through 12 from the text. First, save and run the following file. If need be, change any parameters in the variables.py file to adjust results. Save and rerun variables any time its values are changed.
```bash
python variables.py
```
Now, the following figures can be produced by running the corresponding scripts. All corresponding figures will be saved to your local machine in the subdirectory `figures/`. Each time you run these files, the existing pdf of a figure will be overwritten by the new figure produced.

Figures 2-5 (population size history plot and trajectories of statistics of interest):
```bash
python populationHistoryPlot.py
python statisticTrajectoryPlots.py
```
Figures 6-8 (temporal dynamics of two-locus SFS entries):
```bash
python pTemporalDynamics.py
```
Figure 9 (structure of two-locus SFS at stationarity):
```bash
python SFSSymmetry.py
```
Figure 10 (comparison of ODE solutions to msprime simulations):
```bash
python msprimeSims.py
python odeVsMsprime.py
```
Notice here that the script `msprimeSims.py` with the current number of replicates implemented ($2^{20}$ pairwise replicates and 64 genome replicates with a genome length of 4 * $10^{4}$) will take around two hours to run locally. The script `odeVsMsprime.py` will automatically print the JSD and TVD values comparing the distributions that are included as Table 2 in the text.

Figures 11 and 12 (changing structure of the two-locus SFS with changing mutation and recombination rates):
```bash
python SFSMutRecoStructure.py
```
This file will also automatically output the mutual information values reported in Table 3 in the text.

## Contact

Raunak Kundagrami - rkundagrami@g.harvard.edu

Sean Yetter - smyetter@uchicago.edu

Matthias Steinrücken - steinrue@uchicago.edu
