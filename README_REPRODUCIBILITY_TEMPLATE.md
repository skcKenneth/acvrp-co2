# ACVRP-CO2: Carbon-Aware Asymmetric Vehicle Routing on Real Road Networks

This repository contains code and data for a computational research project on carbon-aware asymmetric capacitated vehicle routing using real OpenStreetMap road networks from Macau and Hong Kong.

The main research question is:

> How much routing, fuel, and CO2 penalty is caused by planning CVRP routes with symmetric distance approximations instead of directed asymmetric road-network matrices?

The project compares four matrix variants:

- **SE**: symmetric Euclidean distance.
- **SM**: symmetric Manhattan distance.
- **SR**: symmetrized road-network distance.
- **AR**: asymmetric road-network distance, treated as ground truth.

All route plans are re-evaluated under AR so that penalties represent performance on the directed road network.

## Repository structure

```text
configs/                 Experiment configuration files
data/                    Fixed and randomized instances, matrices, processed data
graphs/                  Cached OpenStreetMap GraphML files
src/                     Python source code
results/                 Raw and aggregated experiment outputs
figures/                 Generated paper figures
manuscript/tables/       Manuscript-ready LaTeX tables
tests/                   Test suite
```

## Environment setup

Recommended Python version: 3.10 or later.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or, if a Conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate acvrp-co2
```

## Data notes

Road networks are derived from OpenStreetMap using OSMnx. For reproducibility, cached GraphML files should be stored under:

```text
graphs/
```

Randomized customer instances are stored under:

```text
data/randomized/
```

Distance matrices are stored under:

```text
data/matrices/
```

If GraphML files are too large for Git, store them in a release, OSF record, or other stable archive, and document the download link here.

## Reproducing fixed case studies

Run the fixed Macau and Hong Kong case studies with the existing project entry points.

Example commands, to be updated according to the actual scripts:

```bash
python -m src.run_fixed --config configs/macau_fixed.yaml
python -m src.run_fixed --config configs/hongkong_fixed.yaml
```

Expected outputs:

```text
results/fixed/macau_summary.csv
results/fixed/hongkong_summary.csv
results/fixed/solver_grid.csv
```

## Reproducing randomized robustness experiment

### Step 1: Generate randomized instances

```bash
python -m src.random_instances --config configs/randomized.yaml --overwrite
```

Expected outputs:

```text
data/randomized/instances_macau.csv
data/randomized/instances_hongkong.csv
data/randomized/instance_metadata.csv
```

### Step 2: Run randomized OR-Tools experiment

```bash
python -m src.run_randomized --config configs/randomized.yaml --city all --overwrite
```

For a quick smoke test:

```bash
python -m src.run_randomized --config configs/randomized.yaml --city all --max-instances 1 --dry-run
```

Expected outputs:

```text
results/randomized/raw_results.csv
results/randomized/routes/
data/matrices/randomized/
```

### Step 3: Aggregate randomized results

```bash
python -m src.aggregate_results --input results/randomized/raw_results.csv
```

Expected outputs:

```text
results/randomized/penalty_summary.csv
results/randomized/statistical_tests.csv
manuscript/tables/randomized_results.tex
```

### Step 4: Generate figures

```bash
python -m src.make_figures
```

Expected outputs:

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

## Output schemas

### Randomized instance CSV

```text
city,instance_id,node_id,role,osmid,lat,lon,x,y,demand,sampling_seed
```

### Randomized metadata CSV

```text
city,instance_id,num_customers,total_demand,vehicle_capacity,num_vehicles,sampling_seed,status
```

### Raw randomized results CSV

```text
city,instance_id,solver,variant,reference_variant,num_customers,total_demand,distance_m,fuel_l,co2_kg,distance_penalty_pct,fuel_penalty_pct,co2_penalty_pct,feasible,infeasible_reason,runtime_seconds,seed
```

### Aggregated penalty summary CSV

```text
city,variant,n,mean_distance_penalty,sd_distance_penalty,median_distance_penalty,iqr_distance_penalty,mean_fuel_penalty,sd_fuel_penalty,mean_co2_penalty,sd_co2_penalty,median_co2_penalty,iqr_co2_penalty
```

## Reproducibility checklist

Before release, confirm:

- [ ] Fixed case commands run successfully.
- [ ] Randomized instance generation is deterministic.
- [ ] Randomized raw results include all cities, instances, and variants.
- [ ] Infeasible runs, if any, are recorded rather than silently dropped.
- [ ] Aggregated summaries are generated from raw CSV files.
- [ ] Figures are generated from saved result files.
- [ ] Random seeds are recorded.
- [ ] No absolute local paths are hard-coded.
- [ ] Large files are either tracked intentionally or stored externally with links.
- [ ] README commands match actual scripts.

## Known limitations

This project currently focuses on two dense urban cores, synthetic delivery demand, and a linear fuel-emission model. The randomized robustness experiment tests whether fixed-case findings persist across multiple sampled customer sets, but it does not replace real operational delivery data.

Neural solver results should be interpreted as secondary and exploratory unless multiple training seeds and repeated decoding runs are reported.

## Citation

If this repository supports a manuscript or preprint, add citation information here after submission or publication.

```bibtex
@misc{acvrpco2,
  title = {Carbon-Aware Asymmetric Capacitated Vehicle Routing on Real Road Networks},
  author = {Chan, Ka Hin and Cheng, Sok Kin},
  year = {2026},
  note = {Code and data repository}
}
```

## License

Add a license before making the repository public. For code, MIT or BSD-3-Clause are common choices. For data, confirm compatibility with OpenStreetMap attribution requirements.

## OpenStreetMap attribution

This project uses OpenStreetMap data. Any maps or road-network visualizations should include appropriate attribution to OpenStreetMap contributors.
