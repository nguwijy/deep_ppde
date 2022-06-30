# Deep PPDE solver
Authors: Jiang Yu Nguwi and Nicolas Privault.

If this code is used for research purposes, please cite as \
J.Y. Nguwi, G. Penent, and N. Privault.
A deep learning approach to the probabilistic numerical solution of path-dependent partial differential equations.

The paper is available [here](doc/m.pdf).
<br/><br/>

## Quick Start
Deep PPDE solver aims to solve path-dependent PDE.
Since the project involves comparison with other works,
which use different versions of tensorflow,
it is recommended to use Colab to handle package dependencies.
Click the following link for quick start.

* <a href="https://colab.research.google.com/github/nguwijy/deep_ppde/blob/main/comparison.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Explanation
All python files are organized as follows:
1. [main.py](main.py) is our PPDE solver.
2. [regression.py](regression.py) is the PPDE solver based on [[RT17]](#ren2017convergence).
3. [deep_galerkin.py](deep_galerkin.py) is the PPDE solver based on [[SZ20]](#saporito2020pdgm).
4. [ppde_asian.py](deep_signature/ppde_asian.py) and [ppde_barrier.py](deep_signature/ppde_barrier.py) are the PPDE solvers based on [[SVSS20]](#sabate2020solving).
5. [deep_bsde.py](deep_bsde.py) is the PDE solver based on [[HJE18]](#han2018solving).
6. [deep_split.py](deep_split.py) is the PDE solver based on [[BBC+2019]](#beck2019deep).
7. [stats.py](stats.py) summarizes the statistics of the simulations above. It outputs [lit_compare.csv](logs/lit_compare.csv), which can be compared with Table 1-3 in the [paper](doc/m.pdf).


## References
<a id="ren2017convergence">[RT17]</a>
Z. Ren and X. Tan.
On the convergence of monotone schemes for path-dependent PDEs.
*Stochastic Process. Appl.*,
127(6):1738--1762, 2017.

<a id="saporito2020pdgm">[SZ20]</a>
Y.F. Saporito and Z. Zhang.
PDGM: A neural network approach to solve path-dependent partial
differential equations.
Preprint arXiv:2003.02035, 2020.

<a id="sabate2020solving">[SVSS20]</a>
M. Sabate-Vidales, D. Siska, and L. Szpruch.
Solving path dependent PDEs with LSTM networks and path
signatures.
Preprint arXiv:2011.10630, 2020.

<a id="han2018solving">[HJE18]</a>
J. Han, A. Jentzen, and W. E.
Solving high-dimensional partial differential equations using deep
learning.
*Proceedings of the National Academy of Sciences*,
115(34):8505--8510, 2018.

<a id="beck2019deep">[BBC+2019]</a>
C. Beck, S. Becker, P. Cheridito, A. Jentzen, and A. Neufeld.
Deep splitting method for parabolic PDEs.
Preprint arXiv:1907.03452, 2019.
