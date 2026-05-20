# ACVRP-CO₂

**Carbon-Aware Asymmetric Capacitated Vehicle Routing on Real Road
Networks: A Comparative Study of Neural Combinatorial Optimization
and Classical Solvers**

This repository accompanies a high-school research project that
quantifies the cost of ignoring road-network asymmetry in last-mile
delivery, and develops an asymmetric-aware neural combinatorial
optimisation (NCO) policy that the project compares against
state-of-the-art classical solvers (OR-Tools, PyVRP/HGS).

---

## Research Questions

* **RQ1.** When solving the CVRP on real road networks, how much
  extra distance and CO₂ does a *symmetric-distance* baseline incur
  compared to a solver that respects the true *asymmetric* topology?
* **RQ2.** Can a neural policy trained on *synthetic* asymmetric
  instances generalise to *unseen real cities*? How does it compare
  to OR-Tools and to PyVRP's hybrid genetic search?
* **RQ3.** How does the gap scale with instance asymmetry (i.e.,
  the degree of one-way-street prevalence)?

---

## Approach at a glance

**Classical pipeline (Stage 1).**
Four distance-matrix variants (Euclidean, Manhattan, symmetrised road,
asymmetric road) solved with OR-Tools and PyVRP, then all plans
re-evaluated under the asymmetric ground-truth matrix to measure the
*asymmetry penalty*.

**Neural pipeline (Stage 2).**
A custom asymmetric-aware policy network with:

* a **bidirectional edge-attention encoder** that processes the
  road-network arcs in both directions (extends MatNet's dual-graph
  idea from ATSP to ACVRP),
* a **capacity-aware autoregressive decoder** that masks infeasible
  customers and resets capacity on depot returns,
* training via **REINFORCE with the POMO multi-start baseline**
  (no rotation/reflection augmentation, which would corrupt the
  asymmetric matrix),
* edge features include per-arc distance, time, fuel, and CO₂, so
  the model directly attends to the emissions objective.

**Cross-city generalisation study (Stage 3).**
The policy is trained on *synthetic* asymmetric instances and
evaluated on held-out OSM-derived instances from two dense, highly
asymmetric urban centres — **Macau Peninsula** and **Hong Kong
Island Central / Sheung Wan / Wan Chai**. The per-city optimality
gap quantifies how well the model generalises across unseen road
topologies.

---

## Repository layout

```
src/
├── data_loader.py             OSM graph download, customer CSV reader
├── distance_matrix.py         SE / SM / SR / AR distance matrices
├── emissions_model.py         Linear fuel / CO2 model
├── solver_ortools.py          Classical CVRP via OR-Tools
├── solver_ga.py               GA cross-check
├── evaluator.py               Ground-truth re-evaluation on AR
├── visualization.py           Folium + matplotlib outputs
├── experiments.py             Classical experiment runner
├── nco_experiments.py         Neural training + evaluation entry point
├── nco/
│   ├── instance.py            CVRPInstance container + batch collation
│   ├── dataset.py             Synthetic + OSM instance generators
│   ├── encoder.py             Bidirectional edge-attention encoder
│   ├── decoder.py             Capacity-aware pointer decoder
│   ├── model.py               End-to-end ACVRPPolicy
│   ├── trainer.py             REINFORCE+POMO training loop
│   └── inference.py           Greedy / POMO-sampled inference
└── baselines/
    └── solver_pyvrp.py        PyVRP (HGS-CVRP) wrapper

configs/
└── nco_config.yaml            Hyperparameters for the neural pipeline

config.yaml                    Hyperparameters for the classical pipeline (Macau)
config_hongkong.yaml           Alternative city: Hong Kong Central
data/customers_macau.csv       Customer locations -- Macau Peninsula
data/customers_hongkong.csv    Customer locations -- HK Central / Sheung Wan / Wan Chai
docs/methodology.md            Classical formulation, equations, citations
docs/nco_methodology.md        Neural formulation, equations, citations
tests/                         Unit tests (pytest)
notebooks/                     Jupyter quickstart notebooks
models/                        Saved policy checkpoints (gitignored)
```

---

## Installation

Requires **Python 3.10+** and (for the neural pipeline) a CUDA-capable
NVIDIA GPU. CPU-only training is possible but slow.

```bash
git clone https://github.com/<your-username>/acvrp-co2.git
cd acvrp-co2
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If your CUDA version differs from PyTorch's default, install the
matching wheel from <https://pytorch.org/get-started/locally/> before
`pip install -r requirements.txt`.

---

## Reproducing the experiments

### Stage 1 — Classical asymmetry penalty study

```bash
python -m src.experiments --config config.yaml
```

Writes `results/summary.csv`, `results/comparison.png`,
`results/routes_map.html`.

### Stage 2 — Train the neural policies

```bash
# Smoke test (~10 min, verifies the pipeline)
python -m src.nco_experiments --mode train --config configs/nco_config_smoke.yaml

# Full training of MatNet-CVRP (~6 h on RTX 3090 / 4090)
python -m src.train_nco --policy matnet --config configs/train.yaml --osm-eval

# Train the coord-only baseline AM for the comparison (~4 h)
python -m src.train_nco --policy baseline --config configs/train.yaml --osm-eval
```

Best checkpoints are saved to `models/matnet_cvrp_best.pt` and
`models/baseline_am_best.pt`.

### Stage 3 — Cross-city evaluation (NCO only)

```bash
python -m src.nco_experiments \
    --mode eval \
    --config configs/nco_config.yaml \
    --checkpoint models/matnet_cvrp_best.pt
```

Writes `results/nco/nco_eval.json` containing per-instance distances
under each solver, and prints an aggregate gap table.

### Stage 4 — Full 4-solver × 4-matrix grid (the paper's headline experiment)

This produces the comparison table that anchors the manuscript:
OR-Tools, GA, Vanilla-AM, and MatNet-CVRP each run on the SE / SM / SR
/ AR distance matrices for the same customer set, all re-evaluated on
the AR ground truth.

```bash
python -m src.experiments_full \
    --config config.yaml \
    --matnet-checkpoint models/matnet_cvrp_best.pt \
    --baseline-checkpoint models/baseline_am_best.pt
```

Writes `results/summary_full.csv` and `results/penalties_full.json`.

---

## Estimated compute budgets

| Pipeline | Hardware | Wall-clock |
|---|---|---|
| Classical (Stage 1) | CPU, 8 cores | ~10 min |
| NCO smoke test | RTX 3090 | ~10 min |
| NCO full training | RTX 3090 / 4090 | ~6 h |
| Cross-city eval | RTX 3090 + CPU | ~30 min |

---

## Recommended order for a high-school researcher

1. Read `docs/methodology.md`, then `docs/nco_methodology.md`.
2. Run `pytest tests/` to verify the environment.
3. Run the classical Stage 1 — gets a real number for the asymmetry
   penalty without needing a GPU.
4. Run the NCO smoke test to confirm GPU training works.
5. Launch the full MatNet-CVRP training, then go to bed.
6. Train the Vanilla-AM baseline (Stage 2, --policy baseline).
7. Run cross-city evaluation (Stage 3).
8. Run the full grid comparison (Stage 4) — this is the experiment
   whose table goes in the paper.
9. Generate figures and write the manuscript.

---

## License

MIT.
