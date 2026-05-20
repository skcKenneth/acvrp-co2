# Methodology

This document records the mathematical formulation behind the
ACVRP-CO2 project. The notation used here matches the variable names
in the source code so that readers can move between the paper and the
implementation without translation overhead.

## 1. Notation

Let $G = (V, A)$ be a directed graph where $V = \{0, 1, \ldots, n\}$
is the set of customers (with index $0$ reserved for the depot) and
$A \subseteq V \times V$ is the set of feasible arcs. Each customer
$i \in V \setminus \{0\}$ has an integer demand $q_i > 0$, and the
depot has $q_0 = 0$. A homogeneous fleet of $K$ vehicles, each with
capacity $Q$, is based at the depot.

For each arc $(i, j) \in A$ we define a distance $d_{ij}^{(X)}$
according to variant $X \in \{SE, SM, SR, AR\}$ defined below.

## 2. Distance Variants

**SE — Symmetric Euclidean.** The great-circle distance between two
points $(\varphi_i, \lambda_i)$ and $(\varphi_j, \lambda_j)$ in
spherical coordinates is computed via the Haversine formula:

$$
d_{ij}^{SE} = 2 R \arcsin\!\sqrt{\sin^2\!\frac{\varphi_j - \varphi_i}{2}
  + \cos\varphi_i \cos\varphi_j \sin^2\!\frac{\lambda_j - \lambda_i}{2}},
$$

with $R$ the mean Earth radius. Symmetric by construction.

**SM — Symmetric Manhattan.** Project lat/lon onto a local
equirectangular plane centred at the mean latitude $\bar\varphi$, then
take the $L^1$ norm:

$$
d_{ij}^{SM} = |x_i - x_j| + |y_i - y_j|,
$$

with the projection scales chosen so that one degree of latitude or
longitude corresponds to the appropriate metric distance at $\bar\varphi$.

**AR — Asymmetric Road.** Let $G_R = (V_R, A_R)$ be the directed
road-network graph from OSM, with edge weights given by physical
street length. Snap each customer to its nearest node and define

$$
d_{ij}^{AR} = \text{ShortestPath}_{G_R}(\nu(i) \to \nu(j))
$$

where $\nu: V \to V_R$ is the snap mapping. Because $G_R$ honours
one-way streets and turn restrictions, $d_{ij}^{AR} \neq d_{ji}^{AR}$
in general.

**SR — Symmetric Road.** Average the two directions of AR:

$$
d_{ij}^{SR} = \tfrac{1}{2}\bigl(d_{ij}^{AR} + d_{ji}^{AR}\bigr).
$$

## 3. CVRP Decision Model

Let $x_{ij}^k \in \{0, 1\}$ indicate whether vehicle $k$ traverses
arc $(i, j)$. The standard CVRP is the mixed-integer program

$$
\min \sum_{k=1}^{K} \sum_{(i,j) \in A} c_{ij} \, x_{ij}^k
$$

subject to

$$
\sum_{k=1}^{K} \sum_{j \in V} x_{ij}^k = 1
  \qquad \forall\, i \in V \setminus \{0\}
$$

(every customer is visited exactly once),

$$
\sum_{j \in V} x_{ij}^k - \sum_{j \in V} x_{ji}^k = 0
  \qquad \forall\, i \in V,\ k = 1, \ldots, K
$$

(flow conservation per vehicle),

$$
\sum_{i \in V} q_i \sum_{j \in V} x_{ij}^k \le Q
  \qquad \forall\, k
$$

(capacity), plus the customary sub-tour elimination constraints (we
omit the formal MTZ statement here; OR-Tools enforces them implicitly
via its routing model).

## 4. Multi-Objective Arc Cost

The arc cost $c_{ij}$ combines a distance term and a fuel term:

$$
c_{ij} = \alpha_1\, d_{ij} + \alpha_2\, Y_{ij},
$$

where $\alpha_1 + \alpha_2 = 1$ and the fuel burned on an arc is

$$
Y_{ij} = C_1\, d_{ij} + C_2\, t_{ij}
        + C_3\, M_{ij}\, d_{ij}.
$$

Here $t_{ij} = d_{ij} / v$ is the arc-traversal time (with $v$ an
average urban speed), $M_{ij}$ is the payload (kg) the vehicle is
carrying while on the arc, and $C_1, C_2, C_3$ are linear
coefficients calibrated to a light-duty diesel van. Total emissions
are $E = \rho \sum Y_{ij}$ where $\rho$ is the diesel CO₂ factor
(default $2.68$ kg CO₂ per litre).

This linear form is a simplification of the load-aware emissions
model of Bektaş & Laporte (2011); it deliberately drops the
speed-cubed aerodynamic term so the objective remains linear in
$d_{ij}$ and tractable for OR-Tools' arc-cost callback.

## 5. The Asymmetry Penalty

For each variant $X$ we solve the CVRP with arc costs computed from
$d^{(X)}$, obtaining a route plan $\mathcal{R}_X$. We then re-evaluate
$\mathcal{R}_X$ using the ground-truth $d^{(AR)}$ to obtain the
"real" distance $D_X$ and emissions $E_X$. The asymmetry penalty is

$$
\eta_X^{D} = \frac{D_X - D_{AR}}{D_{AR}}, \qquad
\eta_X^{E} = \frac{E_X - E_{AR}}{E_{AR}}.
$$

These two scalars are the headline numbers of the empirical study.

## 6. Solvers

**Exact / heuristic.** OR-Tools' constraint-programming routing
model with `PATH_CHEAPEST_ARC` first-solution heuristic and
`GUIDED_LOCAL_SEARCH` metaheuristic, time-limited to 60 seconds.

**Genetic Algorithm.** Permutation chromosome decoded via Beasley's
split into capacity-feasible routes; ordered crossover (OX1);
position-wise swap mutation; tournament selection with elitism.

## 7. Asymmetry Index

To characterise how asymmetric the AR matrix actually is, define

$$
\mathcal{A} = \frac{\sum_{i < j} \bigl|d_{ij}^{AR} - d_{ji}^{AR}\bigr|}
                   {\sum_{i < j} \bigl(d_{ij}^{AR} + d_{ji}^{AR}\bigr)},
$$

bounded in $[0, 1]$. A value near $0$ means the road network is
practically symmetric; values closer to $1$ indicate one-way streets
dominate.

## 8. References

1. Bektaş, T. & Laporte, G. (2011). The Pollution-Routing Problem.
   *Transportation Research Part B*, 45(8), 1232-1250.
2. Boeing, G. (2017). OSMnx: New methods for acquiring, constructing,
   analyzing, and visualizing complex street networks.
   *Computers, Environment and Urban Systems*, 65, 126-139.
3. Figliozzi, M. A. (2010). Vehicle Routing Problem for Emissions
   Minimization. *Transportation Research Record*, 2197(1), 1-7.
4. Liu, P. et al. (2021). Spatiotemporal-Dependent Vehicle Routing
   Problem Considering Carbon Emissions. *Discrete Dynamics in
   Nature and Society*, 9729784.
5. Perron, L. & Furnon, V. (2024). OR-Tools (Version 9.10) [Software].
   Google. https://developers.google.com/optimization/
6. Fortin, F.-A. et al. (2012). DEAP: Evolutionary Algorithms Made
   Easy. *Journal of Machine Learning Research*, 13, 2171-2175.
