# AGENTS.md

## Project context

This repository supports a student-led computational research project on carbon-aware asymmetric capacitated vehicle routing on real OpenStreetMap road networks.

The main research question is:

> How much routing, fuel, and CO2 penalty is caused by planning CVRP routes with symmetric distance approximations instead of directed asymmetric road-network matrices?

The project currently contains fixed Macau and Hong Kong case studies, multiple matrix variants, classical solvers, neural solvers, and manuscript figures/tables. The next coding goal is to enhance the project into a reproducible computational research package.

The manuscript is maintained separately in Overleaf. Do not edit manuscript files unless explicitly asked. Focus on code, data, results, figures, reproducibility, and documentation.

## Core experiment design

The project has three layers:

### Layer 1: Fixed case studies

- Macau fixed 21-node instance.
- Hong Kong fixed 21-node instance.
- Matrix variants: SE, SM, SR, AR.
- Existing solvers: OR-Tools, GA, MatNet-CVRP, Vanilla-AM.
- Purpose: illustrate the main phenomenon and reproduce the existing tables/figures.

### Layer 2: Randomized robustness experiment

- 30 randomized instances per city by default.
- Each instance has 1 fixed depot and 20 randomly sampled customer nodes.
- Matrix variants: SE, SM, SR, AR.
- Main solver: OR-Tools.
- Every route planned under SE, SM, SR, or AR must be re-evaluated under AR as the ground-truth road network.
- Report distance, fuel, and CO2 penalties relative to the AR-planned route for the same city and instance.
- Purpose: show that the penalty is not only due to hand-picked customer locations.

### Layer 3: Secondary neural solver analysis

- Neural results are secondary.
- Do not overclaim that neural solvers outperform classical solvers.
- Use cautious wording such as “lower sensitivity to matrix choice” or “preliminary robustness”.
- If neural models are stochastic, report seeds or decoding repetitions whenever possible.

## Coding principles

- Do not rewrite the whole project unless explicitly asked.
- Prefer small, reviewable changes.
- Preserve existing working code unless there is a clear bug.
- Add docstrings and type hints to new functions.
- Keep experiment outputs in CSV format.
- Use JSON for saved route plans.
- Use deterministic random seeds.
- Never hard-code absolute local paths.
- Do not silently overwrite generated outputs. Use `--overwrite` when overwriting is intended.
- All generated files should go into `results/`, `figures/`, `data/processed/`, `data/randomized/`, or `data/matrices/`.
- Do not commit large model checkpoints unless explicitly requested.
- Do not delete raw data, graph cache files, or existing results without asking.
- If dependencies are missing, report the missing dependency and the exact command that would have been run.

## Preferred repository structure

Migrate toward this structure only when doing so is safe and does not break existing code:

```text
configs/
  macau_fixed.yaml
  hongkong_fixed.yaml
  randomized.yaml

data/
  fixed/
  randomized/
  matrices/

graphs/
  macau_drive.graphml
  hongkong_drive.graphml

src/
  random_instances.py
  run_randomized.py
  aggregate_results.py
  make_figures.py
  ...existing modules...

results/
  fixed/
  randomized/
  neural/

figures/
  fig1_pipeline.pdf
  fig2_fixed_penalty.pdf
  fig3_randomized_boxplot.pdf
  fig4_solver_heatmap.pdf

manuscript/
  tables/

tests/
```

## Reproducibility requirements

Every experiment row should record, when applicable:

- city
- instance_id
- random_seed or sampling_seed
- solver
- matrix_variant
- reference_variant
- solver_time_limit
- num_customers
- total_demand
- distance_m
- fuel_l
- co2_kg
- distance_penalty_pct
- fuel_penalty_pct
- co2_penalty_pct
- feasible
- infeasible_reason
- runtime_seconds

Every generated CSV should have a documented schema in the README or the script docstring.

## Randomized instance generation requirements

Random customer instances must be reproducible.

For each city:

1. Load the directed OSMnx graph from a GraphML file.
2. Determine or snap the fixed depot to the nearest graph node.
3. Candidate customer nodes must be reachable from the depot and must also be able to reach the depot in the directed graph.
4. Sample without replacement within each instance.
5. Assign synthetic demand using the configured random seed and configured demand range.
6. Save both instance-level and node-level CSV files.

Do not invent depot coordinates. If the config does not contain them and existing fixed configs do not contain them, add a TODO and report it.

## Matrix variants

The project uses four distance matrix variants:

- SE: symmetric Euclidean distance.
- SM: symmetric Manhattan distance.
- SR: symmetrized road-network distance, usually average of directed distances.
- AR: asymmetric road-network distance using directed shortest paths on the OSM graph.

AR is the ground-truth evaluation matrix.

## Evaluation rule

For every city and instance:

1. Solve the route using each matrix variant.
2. Re-evaluate each resulting route under AR.
3. Use the AR-planned route as the reference for penalty calculations.
4. Compute penalties as:

```text
penalty_pct = 100 * (metric_variant_ar_eval - metric_ar_ar_eval) / metric_ar_ar_eval
```

Do not evaluate a route only under the matrix used for planning and call it the final result.

## Testing and validation

After modifying code, run relevant tests or smoke tests.

At minimum, for new randomized experiment code:

```bash
python -m src.random_instances --config configs/randomized.yaml --dry-run
python -m src.run_randomized --config configs/randomized.yaml --max-instances 1 --dry-run
python -m src.aggregate_results --input results/randomized/raw_results.csv
python -m src.make_figures
```

If the project uses pytest, run:

```bash
pytest -q
```

If tests cannot be run, explain why.

## Documentation requirements

When adding or changing scripts, update README or provide a short usage block with:

- purpose of the script
- input files
- output files
- example command
- expected output schema

## Manuscript rules

The manuscript is maintained in Overleaf. Do not modify manuscript files unless explicitly asked.

If asked to generate manuscript-ready outputs, generate:

- `manuscript/tables/*.tex`
- `figures/*.pdf`
- `figures/*.png`
- CSV summaries under `results/`

Do not insert results into `main.tex` unless explicitly requested.

Use cautious research language in any generated text:

Prefer:

- “suggests”
- “indicates”
- “in these case studies”
- “in the randomized instances”
- “under this experimental setup”

Avoid:

- “proves”
- “always”
- “universal”
- “guarantees”
- “must”

## AI disclosure wording

If documentation mentions AI tools, use conservative wording:

> The authors used AI-assisted language editing tools to improve grammar, clarity, and wording. All research questions, modelling choices, code, numerical experiments, figures, results, and interpretations were designed, verified, and approved by the authors.

Do not claim AI generated scientific conclusions, results, data, or code unless explicitly instructed.

## Acceptance criteria for coding tasks

A task is complete only if:

1. The requested files are created or updated.
2. The code runs, or the failure is clearly explained.
3. Output file paths are documented.
4. The schema of every new CSV is documented.
5. Assumptions and TODOs are listed.
6. No unrelated files are modified.
7. No manuscript edits are made unless explicitly requested.
