"""
visualization.py
================

Produces two main outputs:

* An interactive Folium map showing all four route plans overlaid on
  OpenStreetMap tiles, colour-coded by variant.
* A matplotlib bar chart comparing distance / fuel / CO2 across
  variants.

Both outputs are written to the configured results directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox

from .data_loader import Customer
from .evaluator import EvaluatedSolution


# Distinct, colour-blind-aware palette
VARIANT_COLOURS = {
    "SE": "#E69F00",   # orange
    "SM": "#56B4E9",   # sky blue
    "SR": "#009E73",   # bluish green
    "AR": "#D55E00",   # vermilion
}


def _route_polyline(
    graph: nx.MultiDiGraph,
    node_ids: Sequence[int],
    customer_indices: Sequence[int],
) -> List[tuple]:
    """
    Build a list of (lat, lon) points tracing a route through the
    actual road graph rather than straight-line segments.
    """
    points: List[tuple] = []
    for k in range(len(customer_indices) - 1):
        u = node_ids[customer_indices[k]]
        v = node_ids[customer_indices[k + 1]]
        try:
            path = nx.shortest_path(graph, u, v, weight="length")
        except nx.NetworkXNoPath:
            continue
        for node in path:
            data = graph.nodes[node]
            points.append((data["y"], data["x"]))
    return points


def make_folium_map(
    customers: Sequence[Customer],
    graph: nx.MultiDiGraph,
    node_ids: Sequence[int],
    solutions: Dict[str, EvaluatedSolution],
    output_path: str | Path,
) -> None:
    """
    Render all four solutions on one interactive Folium map.

    Each variant becomes its own LayerControl group so the user can
    toggle them independently.
    """
    centre = (customers[0].lat, customers[0].lon)
    m = folium.Map(location=centre, zoom_start=15, tiles="OpenStreetMap")

    # Plot depot and customers
    folium.Marker(
        centre,
        tooltip=f"Depot: {customers[0].name}",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(m)
    for c in customers[1:]:
        folium.CircleMarker(
            (c.lat, c.lon),
            radius=5,
            tooltip=f"{c.name} (demand={c.demand})",
            color="#444",
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    # One FeatureGroup per variant
    for code, ev in solutions.items():
        fg = folium.FeatureGroup(name=f"{code} routes")
        for route in ev.routes:
            poly = _route_polyline(graph, node_ids, route)
            if poly:
                folium.PolyLine(
                    poly,
                    color=VARIANT_COLOURS.get(code, "#888"),
                    weight=4,
                    opacity=0.75,
                    tooltip=f"{code}: {len(route) - 2} stops",
                ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(output_path))


def make_comparison_bar(
    solutions: Dict[str, EvaluatedSolution],
    output_path: str | Path,
) -> None:
    """Bar chart of distance / fuel / CO2 across variants."""
    codes = list(solutions.keys())
    distances = [solutions[c].distance_m / 1000.0 for c in codes]   # to km
    fuels = [solutions[c].fuel_l for c in codes]
    co2s = [solutions[c].co2_kg for c in codes]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colours = [VARIANT_COLOURS.get(c, "#888") for c in codes]

    axes[0].bar(codes, distances, color=colours)
    axes[0].set_ylabel("Distance (km, evaluated on AR)")
    axes[0].set_title("Total tour distance")

    axes[1].bar(codes, fuels, color=colours)
    axes[1].set_ylabel("Fuel (L)")
    axes[1].set_title("Total diesel consumed")

    axes[2].bar(codes, co2s, color=colours)
    axes[2].set_ylabel("CO\u2082 (kg)")
    axes[2].set_title("Total CO\u2082 emissions")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ga_convergence(
    history: Sequence[float],
    output_path: str | Path,
) -> None:
    """Plot best-fitness-per-generation curve for the GA."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="#0072B2")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title("GA convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
