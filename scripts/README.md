




### validate-2.py
This script is used to generate data analyzed in notebook 2. This
includes simulated ARGs from which we record the observed waiting 
distance and predicted waiting distances under the MS-SMC, across
a range of demographic models and Ne values. This is used to validate
the accuracy/error of our approach.

### validate-x5.py
This script is used to generate data that is analyzed in notebook 5. 
This includes simulated ARGs under the SMC' versus Hudson models and
storing their observed waiting distances, and predicted waiting distances
calculated under the MS-SMC. This is used to measure the error/bias in our
calculations caused by the SMC' approximation, and inhomogeneity of trees
between topo-change events.

### mcmc2.py
This script is used to simulate ARGs under a specified species tree and 
then infer the parameters of that species tree using a full likelihood
approach by analyzing (1) genealogy probabilities; (2) waiting distance
probabilities; or (3) both.
