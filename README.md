# waiting-distances
Manuscript and reproducible notebooks for waiting-times manuscript


### Files

manuscript: Latex and PDF files for the manuscript.  
notebooks: Jupyter notebooks with reproducible code (see links below).

### Reproducible notebooks

**Currently requires development versions of toytree and ipcoal, but 
this will be updated to allow for a simpler conda install very soon.**

```bash
# get dependencies from conda
conda install toytree ipcoal -c conda-forge --only-deps

# pip install from github development branches
git clone https://github.com/eaton-lab/toytree -b toy3
cd toytree/
pip install -e . --no-deps

git clone https://github.com/eaton-lab/ipcoal -b toy3
cd ipcoal/
pip install -e . --no-deps

# Will be available soon as:
# conda install ipcoal -c conda-forge
```

#### Links to view notebooks on nbviewer.org

- [notebook 1: Demonstration](https://nbviewer.org/github/eaton-lab/waiting-distances/blob/main/notebooks/nb1-demonstration.ipynb)  
- [notebook 2: Validation](https://nbviewer.org/github/eaton-lab/waiting-distances/blob/main/notebooks/nb2-validations.ipynb)  
- [notebook 3: Likelihood Surface](https://nbviewer.org/github/eaton-lab/waiting-distances/blob/main/notebooks/nb3-likelihood-surface.ipynb)  
- [notebook 4: Likelihood MCMC](https://nbviewer.org/github/eaton-lab/waiting-distances/blob/main/notebooks/nb4-likelihood-mcmc.ipynb)  
- [notebook 4: Topo inhomogeneity bias](https://nbviewer.org/github/eaton-lab/waiting-distances/blob/main/notebooks/nb5-topo-inhomogeneous-bias.ipynb)  

