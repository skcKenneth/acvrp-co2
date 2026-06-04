# Reproducible Computational Research Skill

Use this skill when modifying this repository for computational experiments, reproducibility, result aggregation, or figure generation.

## Purpose

The goal is to make the ACVRP-CO2 project reproducible and suitable for a student research paper. The key improvement is to add a randomized robustness experiment while preserving existing fixed Macau/Hong Kong case studies.

The manuscript itself is maintained in Overleaf. Do not modify manuscript files unless explicitly asked. Instead, generate manuscript-ready tables and figures.

## Standard workflow

For every task:

1. Inspect existing files first.
2. Identify existing functions and entry points before creating new ones.
3. Make the smallest change that satisfies the task.
4. Preserve existing behavior unless fixing a clear bug.
5. Add deterministic seeds for experiments.
6. Save raw results before summaries.
7. Save summaries before figures.
8. Document output schemas.
9. Run a smoke test or explain why it cannot be run.
10. Report changed files and remaining assumptions.

## Output layers

Every new experiment should produce the following layers:

1. Raw results CSV.
2. Aggregated summary CSV.
3. Optional statistical tests CSV.
4. Manuscript-ready LaTeX table under `manuscript/tables/`.
5. Figures under `figures/` in both PDF and PNG formats.
6. Reproduction commands in README or task report.

## File formats

Use:

- CSV for tabular results.
- JSON for routes.
- YAML for configs.
- GraphML for cached OSM graphs.
- PDF and PNG for figures.
- Markdown for documentation.

Do not use binary or notebook-only outputs as the only source of results.

## Randomized robustness experiment specification

Default design:

- Cities: Macau and Hong Kong.
- Instances per city: 30.
- Customers per instance: 20.
- Depot: fixed per city.
- Matrix variants: SE, SM, SR, AR.
- Primary solver: OR-Tools.
- Evaluation: all routes are re-evaluated under AR.
- Reference: AR-planned route for the same city-instance pair.

Required raw result columns:

```text
city,instance_id,solver,variant,reference_variant,num_customers,total_demand,distance_m,fuel_l,co2_kg,distance_penalty_pct,fuel_penalty_pct,co2_penalty_pct,feasible,infeasible_reason,runtime_seconds,seed
```

Required randomized instance columns:

```text
city,instance_id,node_id,role,osmid,lat,lon,x,y,demand,sampling_seed
```

Required metadata columns:

```text
city,instance_id,num_customers,total_demand,vehicle_capacity,num_vehicles,sampling_seed,status
```

## Reachability rule

When sampling directed road-network customer nodes:

- Each customer must be reachable from the depot.
- The depot must be reachable from each customer.
- If the graph has disconnected or one-way unreachable nodes, exclude them from candidate sampling.
- If not enough reachable nodes exist, fail clearly with a useful error message.

## Penalty calculation

For each city-instance-solver:

```text
penalty_pct = 100 * (metric_variant_ar_eval - metric_ar_ar_eval) / metric_ar_ar_eval
```

Where:

- `metric_variant_ar_eval` is the metric of a route planned using SE, SM, SR, or AR and evaluated on AR.
- `metric_ar_ar_eval` is the metric of the route planned using AR and evaluated on AR.

Do not compare routes under different evaluation matrices.

## Aggregation requirements

For each city and non-AR variant, compute:

- n
- mean distance penalty
- standard deviation of distance penalty
- median distance penalty
- IQR distance penalty
- mean fuel penalty
- standard deviation of fuel penalty
- mean CO2 penalty
- standard deviation of CO2 penalty
- median CO2 penalty
- IQR CO2 penalty

If SciPy is available, also compute paired Wilcoxon signed-rank tests comparing each non-AR variant to AR within matched instances. Do not claim significance if the test was not run.

## Figure requirements

Use matplotlib only unless explicitly instructed otherwise.

Generate:

1. `fig1_pipeline.pdf/png`: experimental pipeline diagram.
2. `fig2_fixed_penalty.pdf/png`: fixed-case OR-Tools penalties.
3. `fig3_randomized_boxplot.pdf/png`: randomized CO2 penalty distributions by city and matrix variant.
4. `fig4_solver_heatmap.pdf/png`: fixed-case solver-by-matrix heatmap.

Figures must have readable fonts for a two-column paper.

## Do not do these things

- Do not invent experimental results.
- Do not silently drop infeasible runs.
- Do not hard-code absolute local paths.
- Do not overwrite results without `--overwrite`.
- Do not remove existing fixed-case results.
- Do not delete graph cache files.
- Do not edit Overleaf manuscript files unless explicitly asked.
- Do not claim neural solvers are superior unless the results clearly support it.
- Do not claim universal conclusions from two cities.

## Acceptance criteria

A change is acceptable when:

1. The requested script/config/document exists.
2. The script has a CLI and useful `--help` if applicable.
3. The script runs on a dry-run or small subset.
4. Outputs are saved in the documented locations.
5. Raw results and summary results are separated.
6. Any failed/infeasible run is recorded.
7. The change is documented in README or a task report.
