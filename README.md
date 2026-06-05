# ACVRP-CO2

Carbon-aware asymmetric capacitated vehicle routing on real OpenStreetMap
road networks.

This repository studies how much routing distance, fuel use, and CO2
emissions can change when CVRP routes are planned with symmetric distance
approximations instead of a directed asymmetric road-network matrix. The
current case studies use Macau and Hong Kong road networks from
OpenStreetMap.

The project compares four distance-matrix variants:

- **SE**: symmetric Euclidean distance.
- **SM**: symmetric Manhattan distance.
- **SR**: symmetrized road-network distance.
- **AR**: asymmetric directed road-network distance.

For both the fixed and randomized experiments, route plans are
re-evaluated under **AR** as the ground-truth directed road network. The
reported penalties compare each SE, SM, or SR-planned route against the
AR-planned route for the same instance.

## Repository Structure

```text
config.yaml                         Macau fixed-case configuration
config_hongkong.yaml                Hong Kong fixed-case configuration
configs/randomized.yaml             Randomized robustness configuration
configs/train*.yaml                 Neural solver training configurations

cache/                              Cached OSMnx GraphML road networks
data/customers_*.csv                Fixed 21-node customer sets
data/randomized/                    Generated randomized customer instances
data/matrices/randomized/           Saved randomized distance matrices

src/data_loader.py                  OSM graph loading and customer helpers
src/distance_matrix.py              SE, SM, SR, AR matrix construction
src/solver_ortools.py               OR-Tools CVRP solver
src/solver_ga.py                    Genetic algorithm baseline
src/emissions_model.py              Linear fuel and CO2 model
src/evaluator.py                    AR re-evaluation and penalties
src/experiments.py                  Fixed-case classical runner
src/experiments_full.py             Fixed-case solver-by-matrix runner
src/random_instances.py             Randomized instance generator
src/run_randomized.py               Randomized OR-Tools runner
src/aggregate_results.py            Randomized result aggregation
src/make_figures.py                 Publication figure generation
src/nco/                            Neural solver modules

results_macau/                      Fixed Macau outputs
results_hongkong/                   Fixed Hong Kong outputs
results/randomized/                 Randomized raw results and summaries
figures/                            Generated PDF and PNG figures
tests/                              Pytest test suite
```

## Environment Setup

Recommended Python version: **Python 3.10 or later**.

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The randomized OR-Tools experiment uses the standard scientific and
geospatial stack in `requirements.txt`, including `numpy`, `pandas`,
`networkx`, `osmnx`, `ortools`, `scipy`, and `matplotlib`.

Neural training is optional for the randomized robustness experiment.
The neural modules require PyTorch, and GPU training requires a
CUDA-capable NVIDIA GPU with a compatible PyTorch installation.

To check the basic test suite:

```bash
pytest -q
```

## Reproduce Fixed Case Studies

Run the fixed Macau OR-Tools case study:

```bash
python -m src.experiments --config config.yaml
```

Run the fixed Hong Kong OR-Tools case study:

```bash
python -m src.experiments --config config_hongkong.yaml
```

These write fixed-case summaries, penalty JSON files, maps, and
comparison figures under:

```text
results_macau/
results_hongkong/
```

Run the full fixed-case solver-by-matrix comparison, if the neural
checkpoints are available:

```bash
python -m src.experiments_full --config config.yaml ^
  --matnet-checkpoint models\matnet_cvrp_n20_best.pt ^
  --baseline-checkpoint models\baseline_am_n20_best.pt

python -m src.experiments_full --config config_hongkong.yaml ^
  --matnet-checkpoint models\matnet_cvrp_n20_best.pt ^
  --baseline-checkpoint models\baseline_am_n20_best.pt
```

On macOS/Linux shells, replace the Windows line-continuation character
`^` with `\`.

Expected full-comparison outputs include:

```text
results_macau/summary_full.csv
results_macau/penalties_full.json
results_hongkong/summary_full.csv
results_hongkong/penalties_full.json
```

## Reproduce Randomized Robustness Experiment

The randomized experiment samples 30 instances per city by default.
Each instance has one fixed depot and 20 sampled customer nodes. Customer
nodes are sampled from graph nodes that are reachable from the depot and
can also reach the depot in the directed graph.

Generate randomized instances:

```bash
python -m src.random_instances --config configs/randomized.yaml --overwrite
```

Run the randomized OR-Tools robustness experiment:

```bash
python -m src.run_randomized --config configs/randomized.yaml --overwrite
```

Aggregate randomized results:

```bash
python -m src.aggregate_results --input results/randomized/raw_results.csv
```

Generate publication figures:

```bash
python -m src.make_figures
```

For quick checks without overwriting generated data:

```bash
python -m src.random_instances --config configs/randomized.yaml --dry-run
python -m src.run_randomized --config configs/randomized.yaml --dry-run
python -m src.run_randomized --config configs/randomized.yaml --city macau --max-instances 1 --overwrite
```

## Expected Outputs

Randomized instance files:

```text
data/randomized/instances_macau.csv
data/randomized/instances_hongkong.csv
data/randomized/instance_metadata.csv
```

Randomized experiment outputs:

```text
results/randomized/raw_results.csv
results/randomized/routes/
data/matrices/randomized/
```

Aggregated randomized outputs:

```text
results/randomized/penalty_summary.csv
results/randomized/city_overview.csv
results/randomized/statistical_tests.csv
results/randomized/randomized_results_table.tex
```

Publication figures:

```text
figures/fig1_pipeline.pdf
figures/fig1_pipeline.png
figures/fig2_fixed_penalty.pdf
figures/fig2_fixed_penalty.png
figures/fig3_randomized_boxplot.pdf
figures/fig3_randomized_boxplot.png
figures/fig4_solver_heatmap.pdf
figures/fig4_solver_heatmap.png
```

## Output Schemas

Randomized node-level CSV:

```text
city,instance_id,node_id,role,osmid,lat,lon,x,y,demand,sampling_seed
```

Randomized metadata CSV:

```text
city,instance_id,num_customers,total_demand,vehicle_capacity,num_vehicles,sampling_seed,status
```

Raw randomized results CSV:

```text
city,instance_id,solver,variant,reference_variant,num_customers,total_demand,distance_m,fuel_l,co2_kg,distance_penalty_pct,fuel_penalty_pct,co2_penalty_pct,feasible,infeasible_reason,runtime_seconds,seed
```

Penalty summary CSV:

```text
city,variant,n,mean_distance_penalty,sd_distance_penalty,median_distance_penalty,iqr_distance_penalty,mean_fuel_penalty,sd_fuel_penalty,median_fuel_penalty,iqr_fuel_penalty,mean_co2_penalty,sd_co2_penalty,median_co2_penalty,iqr_co2_penalty,min_co2_penalty,max_co2_penalty
```

City overview CSV:

```text
city,n_instances,mean_ar_distance_m,mean_ar_fuel_l,mean_ar_co2_kg,mean_best_non_ar_co2_penalty,mean_worst_non_ar_co2_penalty
```

## Randomized Results Summary

In these randomized instances, planning with symmetric approximations
produced higher AR-evaluated CO2 than planning directly with AR.

Approximate mean CO2 penalties by city and planning matrix:

| City | SE | SM | SR |
|---|---:|---:|---:|
| Macau | 12.71% | 13.03% | 7.58% |
| Hong Kong | 25.27% | 25.27% | 12.59% |

These values summarize the current `results/randomized/penalty_summary.csv`
file and should be regenerated if the randomized experiment is rerun.

## Data Notes

- Road networks are loaded from cached OSMnx GraphML files under `cache/`.
  The current randomized config uses:
  - `cache/graph_22.1980_113.5430_2000.graphml`
  - `cache/graph_22.2854_114.1577_2000.graphml`
- OpenStreetMap data should be attributed to OpenStreetMap contributors
  in maps, figures, and manuscripts where applicable.
- Randomized customer demand is synthetic. The default randomized config
  samples customer demands from the configured integer range and uses
  depot demand zero.
- Randomized instance generation is deterministic. The base seed is
  defined in `configs/randomized.yaml`.
- Macau instance seeds start at the base seed. Hong Kong instance seeds
  are offset by `100000` in the generated data, so the first Hong Kong
  sampling seed is `100042` when the base seed is `42`.
- Route plans are saved as JSON under `results/randomized/routes/`.
- Randomized matrices are saved under `data/matrices/randomized/` when
  `output.save_matrices` is enabled in `configs/randomized.yaml`.

## Known Limitations

- The empirical case studies cover two cities only: Macau and Hong Kong.
- Randomized customer demand is synthetic rather than observed delivery
  demand.
- The emissions model is a simplified linear fuel and CO2 model.
- The randomized robustness experiment currently uses OR-Tools as the
  primary solver.
- Neural solver comparisons are secondary fixed-case analyses and should
  be interpreted cautiously.
- The results indicate behavior under this experimental setup; they do
  not establish universal routing behavior across all cities or fleets.

## Citation

If you use this repository, please cite the associated manuscript.

## License

This repository is licensed under the MIT License. See `LICENSE` for
details.
