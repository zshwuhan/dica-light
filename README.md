### Rethinking LDA: Moment Matching for Discrete ICA

This project contains Matlab/C++ implementation of the algorithms introduced in the paper
* A. Podosinnikova, F. Bach, and S. Lacoste-Julien. [Rethinking LDA: Moment Matching for Discrete ICA](http://arxiv.org/abs/1507.01784). NIPS, 2015.

Please cite this paper if you use this code for your research.

If you are interested in reproducing our experiments and/or our datasets, check [this](https://github.com/anastasia-podosinnikova/dica) repo.


#### About
This project contains implementations of some moment matching algorithms for topic modeling. 
In brief, these algorithms are based on the construction of moment/cumulant tensors from the data
and matching them to the respective theoretical expressions in order to learn the parameters of the model.

The code consitst of two parts. One part contains the efficient implementation for construction of the moment/cumulant tensors, 
while the other part contains implementations of several so called joint diagonalization type algorithms which are used for matching the tensors. 
Any tensor type (see below) can be arbitrarily combined with one of the diagonalization algorithms (leading, in total, to 6 algorithms).

The focus is on the **latent Dirichlet allocation** (LDA) and **discrete independent component analysis** (DICA) models. 
Importantly, the latter model was shown to be similar and sometimes equivalent to the former.
Respectively, two types of the tensors are considered: the LDA moments and the DICA cumulants. 
The theoretical expressions for the LDA moments were previously derived by Anima Anandkumar, Dean P. Foster, Daniel Hsu, Sham M. Kakade, Yi-Kai Liu 
in A Spectral Algorithm for Latent Dirichlet Allocation. Algorithmica 72(1): 193-214 (2015). 
The expressions for the DICA cumulants are derived in our paper (see below).

The diagonalization type algorithms include the **spectral algorithm** (spectral) based on two eigen decompositions, 
the **orthogonal joint diagonalization algorithm** (jd), and the **tensor power method** (tpm).



#### Quick start

1. make sure your Matlab recognizes C++ compiler: ```mex -setup```
2. save all required paths and build mex files: ```install.m```
3. run examples of all algorithms:  ```examples.m```
7. when finished, remove all paths: ```deinstall.m```


#### Questions?
Please don't hesitate to contact me with questions regarding this code, usage of this algorithm, or bug reports.
