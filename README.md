# Differential_Pricing
This repository contains a Python 3.7 implementation to compare the results of different traffic routing algorithms for a fairness-constrained traffic assignment problem. 

This repository is used in Computation of Interpolated Traffic Assignment Problems (I-TAP), which balance fairness and efficiency, as described in Jalota, Solovey, Zoepf, and Pavone: [“Balancing Fairness and Efficiency in Traffic Routing via Interpolated Traffic Assignment”](https://arxiv.org/abs/2104.00098), *ArXiv*, 2021. In particular, the repository compares the I-TAP solution methodology with the computation of Constrained System Optimum (CSO), which balances fairness and efficiency, as described in Jahn, Möhring, Schulz, and Stier-Moses: “System-Optimal Routing of Traffic Flows with User Constraints in Networks with Congestion“, Operations Research, 2005.

The analysis data (e.g., solution flows, decomposition paths) in this repository is obtained from the repository [git:frank-wolfe-traffic](https://github.com/StanfordASL/frank-wolfe-traffic), whereas the data sets are based on the repository [git:TransportationNetworks](https://github.com/bstabler/TransportationNetworks).
