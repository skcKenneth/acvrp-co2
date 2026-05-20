"""
solver_ga.py
============

A Genetic Algorithm (GA) baseline for the same CVRP instance, intended
as a cross-check against the OR-Tools constraint-programming solver.

Chromosome representation
-------------------------
We use the classical "split into routes" representation:

    Chromosome = a permutation of all customer indices (excluding depot)

To recover routes, we sweep through the chromosome in order, adding
customers to the current route as long as the cumulative demand does
not exceed vehicle capacity; when it does, we close the current route
and open a new one. This is the well-known Beasley split decoder and
makes the search space simple while always producing capacity-feasible
solutions.

Operators
---------
- Selection : tournament selection (size configurable)
- Crossover : ordered crossover (OX) preserves permutation validity
- Mutation  : two-opt swap of two random positions

Fitness
-------
Single objective: alpha_d * distance + alpha_e * 1000 * CO2_kg.
The factor of 1000 brings emissions kg into roughly the same order of
magnitude as distance in metres, so neither swamps the other.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from deap import base, creator, tools

from .emissions_model import (
    EmissionsParams,
    fuel_to_co2_kg,
    route_fuel_litres,
)


# DEAP requires classes to be created at module import time; guard
# against re-creation when re-importing in notebooks.
if not hasattr(creator, "FitnessMinACVRP"):
    creator.create("FitnessMinACVRP", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualACVRP"):
    creator.create("IndividualACVRP", list, fitness=creator.FitnessMinACVRP)


@dataclass
class GASolution:
    routes: List[List[int]]
    objective_value: float
    history: List[float]   # best fitness per generation


def split_into_routes(
    chromosome: Sequence[int],
    demands: Sequence[int],
    vehicle_capacity: int,
    depot: int = 0,
) -> List[List[int]]:
    """
    Decode a permutation chromosome into capacity-feasible routes.

    Walks the chromosome left to right; each time adding the next
    customer would breach capacity, the current route is closed
    (returned to depot) and a new route begins.
    """
    routes: List[List[int]] = []
    current_route = [depot]
    current_load = 0
    for cust in chromosome:
        cust_demand = demands[cust]
        if current_load + cust_demand > vehicle_capacity:
            current_route.append(depot)
            routes.append(current_route)
            current_route = [depot]
            current_load = 0
        current_route.append(cust)
        current_load += cust_demand
    current_route.append(depot)
    routes.append(current_route)
    return routes


def evaluate_chromosome(
    chromosome: Sequence[int],
    distance_matrix: np.ndarray,
    demands: Sequence[int],
    vehicle_capacity: int,
    params: EmissionsParams,
    alpha_distance: float,
    alpha_emissions: float,
    depot: int = 0,
) -> Tuple[float]:
    """Compute the (single-objective) fitness of a chromosome."""
    routes = split_into_routes(chromosome, demands, vehicle_capacity, depot)

    total_distance = 0.0
    total_fuel = 0.0
    for r in routes:
        for k in range(len(r) - 1):
            d = float(distance_matrix[r[k], r[k + 1]])
            if not np.isfinite(d):
                return (1e18,)  # Infeasible
            total_distance += d
        total_fuel += route_fuel_litres(r, distance_matrix, demands, params)

    total_co2 = fuel_to_co2_kg(total_fuel, params)
    fitness = alpha_distance * total_distance + alpha_emissions * 1000.0 * total_co2
    return (fitness,)


# ---------------------------------------------------------------------------
# Crossover and mutation operators
# ---------------------------------------------------------------------------

def ordered_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Classical OX1 ordered crossover used in TSP/VRP literature.

    Selects a random segment from parent1 and fills the remaining
    positions in the order they appear in parent2, skipping elements
    already inherited from parent1.
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))

    def make_child(p_a: List[int], p_b: List[int]) -> List[int]:
        segment = p_a[a:b + 1]
        seen = set(segment)
        remainder = [g for g in p_b if g not in seen]
        # Insert remainder around the preserved segment
        return remainder[:a] + segment + remainder[a:]

    return make_child(parent1, parent2), make_child(parent2, parent1)


def swap_mutation(chromosome: List[int], indpb: float) -> Tuple[List[int]]:
    """
    Each gene is swapped with another random position with probability
    indpb. Preserves permutation validity.
    """
    n = len(chromosome)
    for i in range(n):
        if random.random() < indpb:
            j = random.randrange(n)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return (chromosome,)


# ---------------------------------------------------------------------------
# Main GA driver
# ---------------------------------------------------------------------------

def solve_cvrp_ga(
    distance_matrix: np.ndarray,
    demands: Sequence[int],
    vehicle_capacity: int,
    num_customers: int,
    params: EmissionsParams,
    alpha_distance: float = 0.5,
    alpha_emissions: float = 0.5,
    population_size: int = 120,
    generations: int = 300,
    crossover_prob: float = 0.85,
    mutation_prob: float = 0.15,
    tournament_size: int = 4,
    random_seed: int | None = 42,
    depot: int = 0,
) -> GASolution:
    """
    Run the GA and return the best solution found.

    Parameters
    ----------
    num_customers : int
        Number of customers (excluding depot). Customers are indexed
        1..num_customers in the chromosome.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    toolbox = base.Toolbox()

    customer_indices = list(range(1, num_customers + 1))

    def make_individual():
        perm = customer_indices.copy()
        random.shuffle(perm)
        return creator.IndividualACVRP(perm)

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_chromosome,
        distance_matrix=distance_matrix,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        params=params,
        alpha_distance=alpha_distance,
        alpha_emissions=alpha_emissions,
        depot=depot,
    )
    toolbox.register("mate", lambda a, b: tuple(
        creator.IndividualACVRP(c) for c in ordered_crossover(a, b)
    ))
    toolbox.register("mutate", swap_mutation, indpb=1.0 / num_customers)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    population = toolbox.population(n=population_size)
    # Initial evaluation
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    history: List[float] = []
    for gen in range(generations):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = [creator.IndividualACVRP(list(ind)) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < crossover_prob:
                c1, c2 = toolbox.mate(offspring[i], offspring[i + 1])
                offspring[i], offspring[i + 1] = c1, c2
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_prob:
                (offspring[i],) = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Re-evaluate the changed individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Elitist replacement: best of (parents ∪ offspring)
        combined = population + offspring
        population = tools.selBest(combined, population_size)

        best = tools.selBest(population, 1)[0]
        history.append(best.fitness.values[0])

    best = tools.selBest(population, 1)[0]
    best_routes = split_into_routes(best, demands, vehicle_capacity, depot)
    return GASolution(
        routes=best_routes,
        objective_value=best.fitness.values[0],
        history=history,
    )
