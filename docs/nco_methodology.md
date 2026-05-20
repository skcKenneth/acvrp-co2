# NCO Methodology

This document records the mathematics of the neural component of the
project. It complements `methodology.md` (which covers the classical
ACVRP formulation). Notation is kept identical so the two documents
can be read in sequence.

## 1. Why Neural Combinatorial Optimisation?

Classical solvers such as OR-Tools' guided local search and PyVRP's
hybrid genetic search are excellent on small to medium CVRP instances
but scale poorly: their solve time grows roughly cubically in customer
count, and they must be re-run from scratch whenever a new instance
arrives. Neural Combinatorial Optimisation (NCO) trains a single model
that maps an instance to a route in one neural network forward pass
($O(n^2)$ time, parallelisable on a GPU), which can be many orders of
magnitude faster at inference time.

The price is generalisation: a model trained on synthetic 2-D
Euclidean instances often fails on real, asymmetric, road-network
problems. We therefore design our architecture and training data
specifically with asymmetric road networks in mind.

## 2. Policy Architecture

The policy $\pi_\theta(\boldsymbol{a} \mid \mathbf{x})$ maps an
instance $\mathbf{x}$ (locations, demands, asymmetric distance matrix,
edge-feature tensor) to a probability distribution over routes
$\boldsymbol{a}$. We factor it autoregressively:

$$
\pi_\theta(\boldsymbol{a} \mid \mathbf{x})
  = \prod_{t=1}^{T} \pi_\theta(a_t \mid a_{<t}, \mathbf{x}).
$$

### 2.1 Encoder: Bidirectional Edge Attention

For an asymmetric problem we must let the encoder *see* the difference
between $d_{ij}$ and $d_{ji}$. We do this with two parallel attention
towers over the same node embeddings: one consumes the edge-feature
tensor $\mathbf{E} \in \mathbb{R}^{n \times n \times F}$ in its native
"outgoing" orientation (entry $[i, j]$ describes arc $i \to j$); the
other consumes the transposed tensor $\mathbf{E}^\top$, so its view of
node $i$ is built from the *incoming* arcs.

Within each tower, every layer is a transformer block whose attention
score between query node $i$ and key node $j$ includes an edge bias:

$$
\text{score}(i, j) = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}
  + \mathbf{w}_e^\top \mathbf{e}_{ij},
$$

where $\mathbf{e}_{ij} \in \mathbb{R}^F$ are the per-arc features
(distance, time, fuel, CO₂, all per-instance min-max normalised).

The final node embedding fuses both views with the raw node features:

$$
\mathbf{h}_i = \mathrm{MLP}\bigl([\mathbf{h}_i^{\text{out}};\,
                                   \mathbf{h}_i^{\text{in}};\,
                                   \mathbf{x}_i]\bigr).
$$

This is a deliberate simplification of the dual-graph attention
encoder in MatNet (Kwon et al., 2021), which we extend from ATSP to
ACVRP by enriching the edge-feature dimension and concatenating
demand information into the node features.

### 2.2 Decoder: Capacity-Aware Pointer

At each step $t$ the decoder maintains a state
$\mathbf{s}_t = (\text{current node}, \text{remaining capacity},
\text{visit mask})$. It builds a context vector

$$
\mathbf{c}_t = \mathbf{W}_c\,\bigl[\bar{\mathbf{h}};\;
                                    \mathbf{h}_{a_{t-1}};\;
                                    \tfrac{q^{\text{rem}}_t}{Q}\bigr],
$$

where $\bar{\mathbf{h}} = \tfrac{1}{n}\sum_i \mathbf{h}_i$ is the
graph-mean embedding, $\mathbf{h}_{a_{t-1}}$ is the embedding of the
node we just visited, and $q^{\text{rem}}_t / Q$ is the normalised
remaining capacity.

The next-node distribution is then

$$
\pi_\theta(a_t = j \mid a_{<t}, \mathbf{x}) = \frac{
  \mathbb{1}[j \in \mathcal{F}_t]\;\exp\!\bigl(C \tanh(\mathbf{q}_t^\top
                                                       \mathbf{k}_j)\bigr)
}{\sum_{j' \in \mathcal{F}_t}
  \exp\!\bigl(C \tanh(\mathbf{q}_t^\top \mathbf{k}_{j'})\bigr)},
$$

where $\mathcal{F}_t$ is the set of *feasible* next nodes:

* unvisited customers $j$ with $q_j \le q^{\text{rem}}_t$, and
* the depot, only if the vehicle is not already at it.

The $\tanh$ "clipping" with scale $C$ (typically $C = 10$) prevents
the logits from saturating early in training and is standard since
Bello et al. (2017).

## 3. Training Objective

We train with REINFORCE (Williams, 1992) using the multi-start
baseline of POMO (Kwon et al., 2020):

$$
\nabla_\theta \mathcal{L}(\theta) =
  \mathbb{E}_{\mathbf{x}}\,\frac{1}{N}\sum_{n=1}^{N}
    \bigl(L(\boldsymbol{a}^{(n)}) - \bar{L}(\mathbf{x})\bigr)\,
    \nabla_\theta \log \pi_\theta\bigl(\boldsymbol{a}^{(n)} \mid \mathbf{x}\bigr),
$$

with

$$
\bar{L}(\mathbf{x}) = \frac{1}{N}\sum_{n=1}^{N} L(\boldsymbol{a}^{(n)}).
$$

Each of the $N$ rollouts $\boldsymbol{a}^{(n)}$ is forced to start at
a *different* customer node, which yields a low-variance baseline
without relying on Euclidean rotations or reflections — those
transformations would corrupt the asymmetric matrix and are therefore
invalid for our problem class.

The instance distribution is generated on-the-fly: each batch consists
of fresh randomly perturbed asymmetric instances, ensuring the model
never sees the same instance twice and reducing overfitting.

## 4. Inference Modes

* **Greedy.** Single forward pass with $\arg\max$ at each step.
  Yields one route in $O(n^2)$ GPU time. Used for fast deployment.

* **POMO sampling.** Run $S$ stochastic rollouts from $S$ different
  start customers and return the best (lowest-cost) one. Trades GPU
  time for solution quality; we use $S = 64$ in evaluation.

## 5. Baselines

* **OR-Tools** with Guided Local Search (30 s time budget per instance).
* **PyVRP** (Hybrid Genetic Search, HGS-CVRP), the current SOTA
  classical solver on the CVRPLib benchmarks.

These two anchor the lower end (fast but sub-optimal) and the upper
end (slow but near-optimal) of the classical spectrum.

## 6. Generalisation Study Protocol

The model is trained on synthetic asymmetric instances and never sees
real road networks during training. At evaluation time we sample
held-out instances from five distinct Taiwanese urban areas — Banqiao,
Hsinchu, Kaohsiung, Tainan, Taichung — and report per-city mean cost
plus the optimality gap

$$
\text{gap}_M = \frac{\bar{L}_M - \bar{L}_{\text{PyVRP}}}{\bar{L}_{\text{PyVRP}}}
$$

for each method $M$. Smaller gap means closer to the SOTA classical
solver. The per-city breakdown reveals which topologies the neural
model handles well and which it struggles with.

## 7. References

In addition to the references in `methodology.md`:

1. Bello, I. et al. (2017). Neural Combinatorial Optimization with
   Reinforcement Learning. *arXiv:1611.09940*.
2. Kool, W., van Hoof, H., Welling, M. (2019). Attention, Learn to
   Solve Routing Problems! *ICLR*.
3. Kwon, Y.-D. et al. (2020). POMO: Policy Optimization with Multiple
   Optima for Reinforcement Learning. *NeurIPS*.
4. Kwon, Y.-D. et al. (2021). Matrix Encoding Networks for Neural
   Combinatorial Optimization. *NeurIPS*.
5. Lischka, A. et al. (2024). A GREAT Architecture for Edge-Based
   Graph Problems Like TSP. *arXiv:2408.16717*.
6. Wouda, N. A. et al. (2024). PyVRP: A Vehicle Routing Problem
   Software Suite for Python. *INFORMS Journal on Computing*.
