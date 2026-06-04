# Codex Task Prompts for ACVRP-CO2 Enhancement

Use these prompts one at a time. Do not ask Codex to enhance the entire project in one prompt.

The manuscript is maintained in Overleaf, so these tasks should not edit manuscript text unless explicitly stated. They should generate manuscript-ready tables and figures only.

---

## Task 1 — Repository audit only

```text
You are working on a Python research repository for carbon-aware asymmetric capacitated vehicle routing on OpenStreetMap road networks.

Do not modify files yet.

First, inspect the repository structure and identify:

1. Existing entry points for fixed Macau/Hong Kong experiments.
2. Existing config files.
3. Existing code for:
   - OSM graph loading
   - distance matrix construction
   - OR-Tools solver
   - GA solver
   - emissions evaluation
   - MatNet-CVRP / Vanilla-AM
   - figure generation
4. Existing results files and their schemas.
5. Existing tests.
6. Missing pieces needed for a randomized robustness experiment.

Then produce a concise implementation plan for adding:

- configs/randomized.yaml
- src/random_instances.py
- src/run_randomized.py
- src/aggregate_results.py
- src/make_figures.py
- updated README

Do not change code in this task. Only report findings and propose a plan.
```

---

## Task 2 — Create AGENTS.md

```text
Create a repository-level AGENTS.md file for this project.

The instructions should tell coding agents:

- The project is about carbon-aware asymmetric CVRP on Macau and Hong Kong OpenStreetMap road networks.
- The manuscript is maintained separately in Overleaf, so agents should not edit manuscript files unless explicitly asked.
- The main coding goal is to implement a reproducible randomized robustness experiment.
- Use small reviewable changes.
- Preserve existing fixed-case code.
- Use deterministic random seeds.
- Save raw results, summaries, figures, and route JSONs in documented locations.
- Do not invent results.
- Do not silently overwrite results.
- Do not hard-code absolute local paths.

Also include CSV schemas for randomized instances, metadata, and raw randomized results.
```

---

## Task 3 — Create randomized.yaml

```text
Implement a new randomized robustness experiment configuration.

Create or update:

configs/randomized.yaml

The config should support:

- two cities: macau and hongkong
- fixed depot for each city
- 30 randomized instances per city
- 20 customers per instance
- matrix variants: SE, SM, SR, AR
- OR-Tools as the primary solver
- deterministic seed
- output directories for randomized results, matrices, routes, and figures
- vehicle capacity and number of vehicles
- demand generation settings
- emission model parameters
- OSM graph cache paths

Use existing depot coordinates from the current Macau and Hong Kong fixed configs if available. If not available, add TODO comments instead of inventing coordinates.

Do not run the full experiment.

Also add comments inside the YAML explaining each major block.

After editing, show the final YAML content and explain any assumptions that must be verified manually, such as exact depot coordinates or graph file names.
```

---

## Task 4 — Randomized customer instance generator

```text
Add a randomized customer-set generator for the ACVRP-CO2 project.

Create:

src/random_instances.py

Requirements:

1. Load a directed OSMnx graph from a GraphML file specified in configs/randomized.yaml.
2. Use a fixed depot node from the config or snap the depot coordinates to the nearest graph node.
3. Identify candidate customer nodes that are reachable from the depot and can also reach the depot in the directed graph.
4. For each city, generate N randomized instances, where N comes from the config.
5. Each instance should contain:
   - 1 depot
   - num_customers random customer nodes
   - deterministic random seed
   - generated customer demands
6. Save outputs to:
   data/randomized/instances_<city>.csv
   data/randomized/instance_metadata.csv

CSV schema for instances_<city>.csv:

city,instance_id,node_id,role,osmid,lat,lon,x,y,demand,sampling_seed

CSV schema for instance_metadata.csv:

city,instance_id,num_customers,total_demand,vehicle_capacity,num_vehicles,sampling_seed,status

7. Include a CLI:

python -m src.random_instances --config configs/randomized.yaml

8. Include a --dry-run option that generates only one instance per city and prints summary statistics without overwriting existing files unless --overwrite is passed.

9. Add docstrings and type hints.

10. Do not change existing fixed-case code unless necessary.

After implementation, run a dry-run if dependencies are available. If not, explain the exact command the user should run.
```

---

## Task 5 — Randomized OR-Tools runner

```text
Implement the randomized robustness experiment runner.

Create:

src/run_randomized.py

Requirements:

1. Read configs/randomized.yaml.
2. Load randomized instances from data/randomized/instances_<city>.csv.
3. For each city and instance:
   - construct or load SE, SM, SR, and AR distance matrices
   - solve each matrix variant using OR-Tools
   - re-evaluate every route under the AR matrix
   - compute distance_m, fuel_l, co2_kg
4. Use the AR-optimized solution as the reference for each city-instance pair.
5. Compute:
   - distance_penalty_pct
   - fuel_penalty_pct
   - co2_penalty_pct
6. Save raw long-format results to:
   results/randomized/raw_results.csv

Required CSV schema:

city,instance_id,solver,variant,reference_variant,num_customers,total_demand,distance_m,fuel_l,co2_kg,distance_penalty_pct,fuel_penalty_pct,co2_penalty_pct,feasible,infeasible_reason,runtime_seconds,seed

7. Save routes as JSON to:
   results/randomized/routes/

8. Save matrices if config output.save_matrices is true:
   data/matrices/randomized/

9. Add CLI:

python -m src.run_randomized --config configs/randomized.yaml

10. Add options:
   --city macau|hongkong|all
   --max-instances K
   --overwrite
   --dry-run

11. Make the code robust:
   - if a route is infeasible, record it instead of crashing
   - if a distance matrix has unreachable pairs, record the issue clearly
   - do not silently drop failed instances

12. Do not run the full experiment automatically. Run at most a dry-run or one instance if dependencies are available.

After implementation, print the exact command to run the full experiment.
```

---

## Task 6 — Aggregate randomized results

```text
Create an aggregation script for randomized experiment results.

Create:

src/aggregate_results.py

Requirements:

1. Read results/randomized/raw_results.csv.
2. Exclude AR rows from penalty summaries, because AR is the reference.
3. For each city and variant, compute:
   - n
   - mean_distance_penalty
   - sd_distance_penalty
   - median_distance_penalty
   - iqr_distance_penalty
   - mean_fuel_penalty
   - sd_fuel_penalty
   - mean_co2_penalty
   - sd_co2_penalty
   - median_co2_penalty
   - iqr_co2_penalty
4. Save:
   results/randomized/penalty_summary.csv

5. Also produce a LaTeX table file:
   manuscript/tables/randomized_results.tex

The LaTeX table should report mean ± SD for distance, fuel, and CO2 penalties.

6. If scipy is available, run paired Wilcoxon signed-rank tests comparing each non-AR variant against AR within matched instances. Save:
   results/randomized/statistical_tests.csv

7. Add CLI:

python -m src.aggregate_results --input results/randomized/raw_results.csv

8. Include clear error messages if required columns are missing.

9. Add a small internal validation:
   - check that each city-instance has one AR reference row
   - check that penalty columns are numeric
   - check that no successful non-AR rows are missing reference data
```

---

## Task 7 — Generate figures

```text
Create publication-quality figure generation scripts.

Create or update:

src/make_figures.py

Inputs:
- existing fixed-case results if available
- results/randomized/raw_results.csv
- results/randomized/penalty_summary.csv

Outputs:
- figures/fig1_pipeline.pdf
- figures/fig1_pipeline.png
- figures/fig2_fixed_penalty.pdf
- figures/fig2_fixed_penalty.png
- figures/fig3_randomized_boxplot.pdf
- figures/fig3_randomized_boxplot.png
- figures/fig4_solver_heatmap.pdf
- figures/fig4_solver_heatmap.png

Requirements:

1. Use matplotlib only.
2. Do not use seaborn.
3. Use readable font sizes suitable for a two-column paper.
4. Save both PDF and PNG versions.
5. Figure 1 should be a simple experiment pipeline diagram:
   OSM graph -> customer instances -> SE/SM/SR/AR matrices -> solvers -> AR re-evaluation -> penalty metrics.
6. Figure 2 should show fixed-case OR-Tools penalties for Macau and Hong Kong.
7. Figure 3 should show randomized CO2 penalty distributions by city and matrix variant using boxplots.
8. Figure 4 should show a solver-by-matrix heatmap for fixed-case penalties, with separate panels for Macau and Hong Kong if possible.
9. Add CLI:

python -m src.make_figures

10. If an input file is missing, skip that figure and print a warning instead of crashing.

Do not edit manuscript/main.tex.
```

---

## Task 8 — README reproducibility guide

```text
Rewrite the README as a reproducibility guide for the paper.

Requirements:

1. Start with a short overview of the research question.
2. Add environment setup instructions:
   - Python version
   - pip install -r requirements.txt
   - or conda env create -f environment.yml
3. Add commands to reproduce:
   - fixed case studies
   - randomized instance generation
   - randomized OR-Tools experiment
   - aggregation
   - figure generation
4. Add expected output files.
5. Add repository structure.
6. Add notes on data:
   - OpenStreetMap graph cache
   - GraphML files
   - randomized instances
   - distance matrices
7. Add citation section.
8. Add license section.
9. Add a Known limitations section.

Do not include unsupported claims about publication status.
Do not include private file paths.
Do not edit manuscript/main.tex.
```

---

## Task 9 — Final QA pass

```text
Perform a final reproducibility QA pass.

Do not make large changes unless necessary.
Do not edit manuscript/main.tex.

Check:

1. Can a new user reproduce fixed results from README commands?
2. Can a new user reproduce randomized results from README commands?
3. Are all generated CSV files documented?
4. Are all manuscript figures generated from code?
5. Are all manuscript-ready tables generated from CSV or LaTeX table files?
6. Are there any hard-coded local paths?
7. Are random seeds recorded?
8. Are large files excluded from git unless required?
9. Does pytest pass, or are failures explained?
10. Are graph cache files or instructions available?
11. Are failed/infeasible runs recorded instead of silently dropped?
12. Are neural solver claims cautious and secondary in documentation?

Produce a QA report with:

- PASS items
- FAIL items
- suggested fixes
- files that need manual review

Do not rewrite files unless fixing a clear error.
```
