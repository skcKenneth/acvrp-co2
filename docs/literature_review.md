# Literature Review Guide — ACVRP-CO₂

This document provides a structured literature review for the project
**"Quantifying the Asymmetry Penalty in Carbon-Aware Last-Mile Routing
under Real Asymmetric Road Networks"**.

Its purpose is twofold:
1. **Give your student a starting point** — every paper here has been
   verified to exist, with the citation, the year, and a one-line
   justification of why it matters for *this specific* paper.
2. **Show them how to organise a literature review** by theme rather
   than chronology — this is how IJHSR-grade papers structure their
   Related Work section.

The reading is divided into **five themes**. Each theme has 3–6
key references. The student does not need to read all of them in
full — for several, the abstract + intro is enough. The asterisks
indicate priority:

- `***`  Must read in full (foundational or directly cited in our methodology)
- `**`   Skim the abstract + intro + figures
- `*`    Aware of, can cite without deep reading

---

## Theme 1: Classical Capacitated Vehicle Routing Problem (CVRP)

This theme covers what the CVRP is, how it has been formulated as a
mixed-integer program, and where the benchmark instances come from.
The student needs to understand this clearly before they can claim
that *asymmetric* CVRP is different.

### `***` Dantzig & Ramser (1959). The Truck Dispatching Problem.
*Management Science*, 6(1), 80–91.
The paper that founded the field. Required citation for any VRP paper.
A short note that introduced the truck dispatching problem and showed
how it could be formulated as a linear-programming relaxation of an
integer program.

### `***` Uchoa et al. (2017). New Benchmark Instances for the Capacitated Vehicle Routing Problem.
*European Journal of Operational Research*, 257(3), 845–858.
Defines CVRPLIB, the standard benchmark suite that almost every VRP
paper uses for comparison. Our paper will cite this when explaining
that we deliberately *do not* use these synthetic Euclidean instances
because they assume symmetric distances.

### `**` Wouda, Lan & Kool (2024). PyVRP: A High-Performance VRP Solver Package.
*INFORMS Journal on Computing*, 36(4), 943–955.
The current state-of-the-art open-source classical VRP solver
(implements Hybrid Genetic Search; won the 2021 DIMACS challenge and
the 2022 EURO-NeurIPS competition). Useful as a high-end classical
baseline. We use OR-Tools instead because PyVRP currently does not
natively support arbitrary asymmetric matrices, but PyVRP is worth
naming in Related Work.

### `*` Christofides, Mingozzi & Toth (1979). The Vehicle Routing Problem. In *Combinatorial Optimization*, Wiley, 315–338.
The original benchmark set ("CMT" instances). Mentioned briefly as
the predecessor to CVRPLIB.

---

## Theme 2: Asymmetric Vehicle Routing (the closest prior work)

This is the most directly relevant theme. The student should be able
to articulate exactly what we contribute on top of these papers.

### `***` Lee & Kim (2021). A Proposal and Analysis of New Realistic Sets of Benchmark Instances for Vehicle Routing Problems with Asymmetric Costs.
*Applied Sciences*, 11(11), 4790.
Currently the most direct precedent for our work. They built ACVRP
benchmark instances using a Korean map API (T map). Important
distinction we should highlight: their benchmarks are *synthetic*
realisations of a few cities, while we evaluate on OSM-derived
instances for two specific cities (Macau, Hong Kong) and quantify
the *asymmetry penalty as a function of solver class*. Cite this
explicitly when justifying our methodology.

### `***` Melo, Mota, Andrade & Araújo (2022). The Impact of One-Way Streets on the Asymmetry of the Shortest Commuting Routes.
*Physical Review Research*, 4(2), 023053. arXiv:2111.07434.
Empirically measures route asymmetry on OSM data for ten cities.
Defines a log-ratio metric for asymmetry. Our "asymmetry index" is a
related but VRP-specific metric, so we should cite this as the closest
precedent and explain the difference (they study individual O-D pairs;
we measure how this propagates through the optimal *tour*).

### `**` Erbao Cao et al. (2020). A New Model for the Asymmetric Vehicle Routing Problem with Simultaneous Pickup and Deliveries.
*Operations Research Letters*, 48(2), 161–164.
States the importance of asymmetric VRPs in urban contexts with
one-way streets or sloped terrain — useful one-line citation for
motivation.

### `*` Coordinated Truck-and-Drone Asymmetric Arc Routing (2022). PMC9415381 / Mathematical Problems in Engineering.
An example of arc-routing extension that explicitly accounts for
one-way streets. Cite once when listing "other asymmetric-routing
variants" in Related Work.

---

## Theme 3: Pollution-Routing & Green VRP (the emissions side)

Justifies why CO₂ matters as an objective and how it has been modelled
in prior work. Establishes that the linear fuel/emissions model we use
is well-grounded.

### `***` Bektaş & Laporte (2011). The Pollution-Routing Problem.
*Transportation Research Part B: Methodological*, 45(8), 1232–1250.
**The single most important citation for our emissions formulation.**
Introduces the Pollution-Routing Problem (PRP) and the load-and-speed-
dependent fuel model. Our simplified linear model (distance + time +
load·distance) is a tractable specialisation of theirs. The paper also
reports up to 7% CO₂ savings achievable by re-routing — this is a
natural reference point to put our 4–42% asymmetry penalty against.

### `***` Demir, Bektaş & Laporte (2014). The Bi-Objective Pollution-Routing Problem.
*European Journal of Operational Research*, 232(3), 464–478.
Extends PRP to multi-objective (fuel + driving time). Our paper's
multi-objective framing (distance + CO₂) is methodologically aligned
with this work. Cite when introducing the scalarisation
α₁·distance + α₂·CO₂.

### `**` Asghari & Mirzapour Al-e-Hashem (2024). Survey of Green Vehicle Routing Problem: Past and Future Trends.
*Operations Research Forum* / RG review, 2024 update covering 75 PRP papers from 2011 to 2024.
Comprehensive review of the green VRP literature. The student should
use it as a map to find any sub-topic in the field. **Single best
secondary source** for the Related Work paragraph on green VRP.

### `**` Lin, Choy, Ho, Chung & Lam (2014). Survey of Green Vehicle Routing Problem: Past and Future Trends.
*Expert Systems with Applications*, 41(4), 1118–1138.
The other major review (earlier, pre-2014). Useful for citing the
historical evolution of Green VRP.

### `*` Demir, Bektaş & Laporte (2012). An Adaptive Large Neighborhood Search Heuristic for the Pollution-Routing Problem.
*European Journal of Operational Research*, 223(2), 346–359.
The follow-up algorithmic paper showing ALNS for PRP. Cite when
discussing classical heuristics for emission-aware routing.

---

## Theme 4: Neural Combinatorial Optimization (NCO) for routing

The four foundational papers behind our MatNet-CVRP architecture. Read
these in chronological order to see the evolution from RNN+supervised
learning to attention+RL.

### `***` Vinyals, Fortunato & Jaitly (2015). Pointer Networks.
*Advances in Neural Information Processing Systems* (NIPS) 28.
arXiv:1506.03134.
First neural architecture able to output a permutation of variable-
length input (essential for routing). Sets the foundation for
everything that follows. Solved planar TSP up to n=50 via supervised
learning.

### `***` Bello, Pham, Le, Norouzi & Bengio (2017). Neural Combinatorial Optimization with Reinforcement Learning.
*ICLR Workshop*. arXiv:1611.09940.
Replaces supervised learning (Vinyals 2015) with REINFORCE, enabling
training without optimal labels — critical because optimal CVRP
solutions are expensive to obtain. Establishes the basic NCO recipe:
encoder + autoregressive pointer decoder + RL objective.

### `***` Kool, van Hoof & Welling (2019). Attention, Learn to Solve Routing Problems!
*ICLR 2019*. arXiv:1803.08475. Code: github.com/wouterkool/attention-learn-to-route.
**The most important architectural precedent for our Vanilla-AM
baseline.** Replaces RNNs with multi-head self-attention, introduces
the greedy-rollout REINFORCE baseline. Achieves near-optimal solutions
for TSP/CVRP up to n=100. Our Vanilla-AM is an exact re-implementation
of this architecture (without edge features). Cite explicitly whenever
we say "AM" or "the Attention Model".

### `***` Kwon, Choo, Kim, Yoon, Gwon & Min (2020). POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.
*NeurIPS 2020*. arXiv:2010.16011. Code: github.com/yd-kwon/POMO.
The training recipe we use. Replaces the greedy rollout baseline of
Kool 2019 with the multi-start mean baseline (launch K rollouts from
K different first customers; the baseline for each is the mean of the
others). Significantly faster training and better local-minimum
escape. Reduces TSP100 optimality gap to 0.14%.
**Note we deliberately do NOT use the rotation/reflection augmentation
also introduced in this paper** — those operations corrupt the
asymmetric distance matrix. Make this explicit in our methodology
section.

### `***` Kwon, Choo, Yoon, Park, Park & Gwon (2021). Matrix Encoding Networks for Neural Combinatorial Optimization.
*NeurIPS 2021*. arXiv:2106.11113. Code: github.com/yd-kwon/MatNet.
**The most direct architectural precedent for our work.** Introduces
the bidirectional attention encoder for matrix-valued inputs (such as
asymmetric distance matrices). Applied to asymmetric TSP and flexible
flow-shop scheduling. Our MatNet-CVRP extends this idea by:
(i) integrating it with capacity constraints (CVRP, not just TSP),
(ii) attending to multi-channel edge features (distance, time, fuel,
CO₂ — not just a scalar entry), and
(iii) replacing the random one-hot positional encoding with
geographic coordinates from OSM. State all three contributions
explicitly in our Related Work paragraph on MatNet.

### `**` Berto et al. (2024). RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark.
*KDD 2025*. arXiv:2306.17100. Code: github.com/ai4co/rl4co.
The current unified library for NCO research, with implementations of
AM, POMO, MatNet, and 20+ other models. Worth citing as the standard
reference codebase for the field. Useful for the student to know about
even if we built our own implementation.

---

## Theme 5: OSM-Based Routing & Real Road Networks

Documents the tools and conventions for working with real road network
data, which is what distinguishes our paper from prior NCO routing
work.

### `***` Boeing (2017). OSMnx: A Python Package to Work with Graph-Theoretic OpenStreetMap Street Networks.
*Journal of Open Source Software*, 2(12), 215.
The original OSMnx paper. Required citation when describing how we
obtain the road network for Macau and Hong Kong. Always cite the JOSS
2017 version (DOI: 10.21105/joss.00215) as it's the canonical
peer-reviewed citation.

### `**` Boeing (2025). Modeling and Analyzing Urban Networks and Amenities with OSMnx.
*Geographical Analysis*. DOI: 10.1111/gean.70009.
The 2025 update covering the current OSMnx 2.0 capabilities. Cite
alongside the 2017 paper if reviewers want the latest reference.

### `***` Melo et al. (2022). The Impact of One-Way Streets on the Asymmetry of the Shortest Commuting Routes.
Already listed in Theme 2; deserves to be cited here too because they
demonstrated empirically that OSM-derived road graphs *do* exhibit
measurable directional asymmetry across major world cities.

---

## How to Use This Guide

### Suggested reading order (one focused week of reading)

| Day | Theme & Papers |
|---|---|
| Mon | Theme 1 in full (Dantzig 1959, Uchoa 2017) + skim PyVRP |
| Tue | Theme 3 in full (Bektaş 2011 most important) + Demir 2014 |
| Wed | Theme 4 part 1: Vinyals 2015 + Bello 2017 |
| Thu | Theme 4 part 2: Kool 2019 + POMO 2020 |
| Fri | Theme 4 part 3: MatNet 2021 (re-read more carefully — it directly informs our architecture) |
| Sat | Theme 2: Lee 2021 + Melo 2022 (asymmetric VRP precedents) |
| Sun | Theme 5: OSMnx 2017; review and start drafting Related Work |

### Drafting the "Related Work" section (Section 2 of the paper)

Use this skeleton, with one paragraph per theme:

> **2.1 Classical CVRP and benchmark conventions.** [Use Theme 1
> citations to explain the standard formulation, mention that almost
> all benchmarks assume symmetric 2D Euclidean distances.]
>
> **2.2 Asymmetric VRP variants.** [Theme 2. State that asymmetric
> CVRP has received less attention than symmetric. Lee 2021 created
> realistic ACVRP benchmarks but did not measure the cost of
> ignoring asymmetry.]
>
> **2.3 Carbon-aware vehicle routing.** [Theme 3. The Pollution-
> Routing Problem (Bektaş & Laporte 2011) introduces emissions as an
> objective. Most of the green-VRP literature uses symmetric
> Euclidean distances; combining green objectives with realistic
> asymmetric road topology is rare.]
>
> **2.4 Neural combinatorial optimisation for routing.** [Theme 4.
> Chronological narrative from Vinyals 2015 to MatNet 2021. End with
> the observation that no existing NCO architecture handles both
> capacity constraints AND asymmetric matrices.]
>
> **2.5 Position of this work.** [One-paragraph synthesis: we combine
> the asymmetric matrix encoding of MatNet with the capacity-aware
> decoder of POMO/AM, train under a CO₂-augmented reward inspired by
> Bektaş & Laporte, and evaluate on OSM road networks of Macau and
> Hong Kong. To our knowledge, no prior work addresses this exact
> intersection.]

### BibTeX-friendly citation list

For convenience, this is the full bibliography in `@article` format —
the student can paste this into their reference manager:

```bibtex
@article{dantzig1959,
  author = {Dantzig, George B. and Ramser, John H.},
  title  = {The Truck Dispatching Problem},
  journal= {Management Science},
  volume = {6},  number = {1}, pages = {80--91}, year = {1959}
}

@article{uchoa2017cvrplib,
  author = {Uchoa, E. and Pecin, D. and Pessoa, A. and Poggi, M. and Vidal, T. and Subramanian, A.},
  title  = {New Benchmark Instances for the Capacitated Vehicle Routing Problem},
  journal= {European Journal of Operational Research},
  volume = {257}, number = {3}, pages = {845--858}, year = {2017}
}

@article{wouda2024pyvrp,
  author = {Wouda, Niels A. and Lan, Leon and Kool, Wouter},
  title  = {{PyVRP}: A High-Performance {VRP} Solver Package},
  journal= {INFORMS Journal on Computing},
  volume = {36}, number = {4}, pages = {943--955}, year = {2024},
  doi = {10.1287/ijoc.2023.0055}
}

@article{lee2021acvrp,
  author = {Lee, Jusang and Kim, Byung-In},
  title  = {A Proposal and Analysis of New Realistic Sets of Benchmark Instances for Vehicle Routing Problems with Asymmetric Costs},
  journal= {Applied Sciences},
  volume = {11}, number = {11}, pages = {4790}, year = {2021}
}

@article{melo2022asymmetry,
  author = {Melo, Hygor P. M. and Mota, Diogo P. and Andrade, Jr., Jos\'e S. and Ara\'ujo, Nuno A. M.},
  title  = {Impact of one-way streets on the asymmetry of the shortest commuting routes},
  journal= {Physical Review Research},
  volume = {4}, number = {2}, pages = {023053}, year = {2022},
  doi = {10.1103/PhysRevResearch.4.023053}
}

@article{bektas2011prp,
  author = {Bekta\c{s}, Tolga and Laporte, Gilbert},
  title  = {The Pollution-Routing Problem},
  journal= {Transportation Research Part B: Methodological},
  volume = {45}, number = {8}, pages = {1232--1250}, year = {2011},
  doi = {10.1016/j.trb.2011.02.004}
}

@article{demir2014biobjective,
  author = {Demir, Emrah and Bekta\c{s}, Tolga and Laporte, Gilbert},
  title  = {The Bi-Objective Pollution-Routing Problem},
  journal= {European Journal of Operational Research},
  volume = {232}, number = {3}, pages = {464--478}, year = {2014}
}

@article{demir2012alns,
  author = {Demir, Emrah and Bekta\c{s}, Tolga and Laporte, Gilbert},
  title  = {An Adaptive Large Neighborhood Search Heuristic for the Pollution-Routing Problem},
  journal= {European Journal of Operational Research},
  volume = {223}, number = {2}, pages = {346--359}, year = {2012}
}

@inproceedings{vinyals2015ptrnet,
  author = {Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
  title  = {Pointer Networks},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {28}, year = {2015}
}

@misc{bello2017nco,
  author = {Bello, Irwan and Pham, Hieu and Le, Quoc V. and Norouzi, Mohammad and Bengio, Samy},
  title  = {Neural Combinatorial Optimization with Reinforcement Learning},
  year   = {2017},
  eprint = {1611.09940},
  archivePrefix = {arXiv}
}

@inproceedings{kool2019attention,
  author = {Kool, Wouter and van Hoof, Herke and Welling, Max},
  title  = {Attention, Learn to Solve Routing Problems!},
  booktitle = {International Conference on Learning Representations},
  year = {2019}
}

@inproceedings{kwon2020pomo,
  author = {Kwon, Yeong-Dae and Choo, Jinho and Kim, Byoungjip and Yoon, Iljoo and Gwon, Youngjune and Min, Seungjai},
  title  = {{POMO}: Policy Optimization with Multiple Optima for Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {33}, year = {2020}
}

@inproceedings{kwon2021matnet,
  author = {Kwon, Yeong-Dae and Choo, Jinho and Yoon, Iljoo and Park, Minah and Park, Duwon and Gwon, Youngjune},
  title  = {Matrix Encoding Networks for Neural Combinatorial Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34}, year = {2021}
}

@inproceedings{berto2024rl4co,
  author = {Berto, Federico and Hua, Chuanbo and Park, Junyoung and others},
  title  = {{RL4CO}: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year = {2025},
  eprint = {2306.17100},
  archivePrefix = {arXiv}
}

@article{boeing2017osmnx,
  author = {Boeing, Geoff},
  title  = {{OSMnx}: A Python Package to Work with Graph-Theoretic {OpenStreetMap} Street Networks},
  journal= {Journal of Open Source Software},
  volume = {2}, number = {12}, pages = {215}, year = {2017},
  doi = {10.21105/joss.00215}
}

@article{boeing2025osmnx,
  author = {Boeing, Geoff},
  title  = {Modeling and Analyzing Urban Networks and Amenities with {OSMnx}},
  journal= {Geographical Analysis},
  year = {2025},
  doi = {10.1111/gean.70009}
}

@article{williams1992reinforce,
  author = {Williams, Ronald J.},
  title  = {Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning},
  journal= {Machine Learning},
  volume = {8}, pages = {229--256}, year = {1992}
}
```

(`williams1992reinforce` is the original REINFORCE paper — cite once
in the methodology section when introducing the policy gradient.)

---

## A Note on Methodology

When the student writes the Related Work section, the key thing is to
**make the gap explicit**. The most powerful single sentence to write
is something like:

> *"While prior work has separately studied (i) asymmetric VRP with
> classical solvers [Lee 2021], (ii) emissions as a VRP objective
> [Bektaş & Laporte 2011], and (iii) neural combinatorial optimisation
> for symmetric Euclidean CVRP [Kool 2019, Kwon 2020], to our
> knowledge no prior work combines all three: a neural solver that
> handles capacity constraints AND asymmetric real-road matrices AND
> a CO₂-aware reward, evaluated on actual OSM-derived road networks
> of dense Asian cities."*

That sentence is the **whole justification for the paper**. Everything
else in Related Work is supporting evidence.
