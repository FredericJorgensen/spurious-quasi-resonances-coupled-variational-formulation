# Spurious Quasi-Resonances for Stabilized BIE-Volume Formulations for the Helmholtz Transmission Problem

This github repository contains all the code used to write the report *Spurious Quasi-Resonances for Stabilized BIE-Volume Formulations for the Helmholtz Transmission Problem* (contained in the report directory) under supervision of Prof. Hiptmair (ETH Zurich).

## Abstract
A particular *regularised variational formulation* of the Helmholtz transmission problem is studied on a two-dimensional disk for varying frequencies. In particular, the operator norm of its associated inverse operator is investigated.
In scenarios where the inner refractive index is bigger than the outer one, the operator norm associated with the considered formulation resonates and grows as a function of the wave number. The solution operator did not expose this resonance and growth behavior. This behavior, studied in previous research for other operators, is called *spurious quasi-resonances*. The occurring resonances, however, were much weaker in the formulation considered here than in others. The origin of these resonances  is explained.
We consider the Helmholtz transmissisolve a particular

## File Structure 
- figures: contains all figures generated in the code, structured by subfolder.
- report: contains the report, the *Latex* files and all figures used to generate it
- python files im main directory: contain all simulations, plot generators, and validations used in the paper.

## Run the code
To run all code that was for the report, execute `main.py`. The Helmholtz and numerical parameters can be adjusted in the same file.