# Target Stationary Distribution Problem (TSDP)
### Linear Programming for Editing Markov Chains on Wikipedia Clickstream Graphs

> Given a real-world Markov chain built from Wikipedia clickstream data, what is the **minimal set of edits** to its transition probabilities that forces its long-run behaviour to match a target distribution? This project formulates that question as a **sparse Linear Program**, builds an end-to-end data pipeline over the October 2025 English Wikipedia Clickstream dump, and scales the solution with a **Column Generation** scheme.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue">
  <img alt="SciPy" src="https://img.shields.io/badge/Solver-SciPy%20HiGHS-orange">
</p>

---

## Overview

A **stationary distribution** $\pi$ of a Markov chain describes where a random surfer spends time in the long run. On a Wikipedia clickstream graph, this distribution is highly *skewed*: a handful of hub pages absorb most of the long-run attention.

**TSDP** asks the inverse-design question: how do we nudge the transition matrix so that the long-run attention follows a *desired* target $\pi^\*$ (e.g. a flat, uniform distribution that de-biases popularity) — while changing the original chain **as little as possible**?

The answer is cast as a convex **Linear Program** minimizing the entrywise $\ell_1$ distance between the edited matrix and the baseline, subject to keeping the result a valid Markov chain whose stationary distribution equals $\pi^\*$.

---

## Problem Statement

Given a baseline row-stochastic matrix $\tilde{P}$ derived from clickstream data and a desired stationary distribution $\pi^\*$, find a new transition matrix $P$ that is **as close as possible** to $\tilde{P}$ while satisfying the stationarity constraint $\pi^{*\top} P = \pi^{*\top}$, and remaining row-stochastic and nonnegative.

The distance between $P$ and $\tilde{P}$ is measured with the entrywise $\ell_1$ norm:

$$
\|P - \tilde{P}\|_1 = \sum_{i,j} \lvert P_{ij} - \tilde{P}_{ij} \rvert .
$$

The core optimization problem is therefore:

$$
\begin{aligned}
\min_{P} \quad & \sum_{(i,j)\in\Omega} c_{ij}\,\lvert P_{ij} - \tilde{P}_{ij}\rvert \\
\text{s.t.} \quad & \sum_{j} P_{ij} = 1, \quad \forall i, \\
& P_{ij} \ge 0, \\
& \pi^{*\top} P = \pi^{*\top},
\end{aligned}
$$

where $\Omega$ is the set of editable edges. In the *full-edges* variant $\Omega = V \times V$; in practice we restrict $\Omega$ to a sparse, data-driven support (see below).

---

## Data Pipeline

The pipeline turns the raw **English Wikipedia Clickstream (October 2025)** dump into topic-specific, self-contained subgraphs.

### 1. Raw data & initial cleaning
- Load `clickstream-enwiki-2025-10.tsv` with the four standard fields: `prev` (source page), `curr` (destination page), `type` (transition type), `n` (click count).
- Drop rows with missing values (only **128** rows out of **36.5M**).
- Keep only intra-Wikipedia hyperlink transitions (`type = "link"`) and discard pages prefixed with `other-` (e.g. `other-search`, `other-external`).
- **Result:** the dataset shrinks from **36.5M → 32M** rows.

### 2. Topic-specific subgraph extraction
- Match pages by **case-insensitive substring** on the topic $t$:

$$
\text{mask}(t) = (\texttt{prev} \ni t) \;\lor\; (\texttt{curr} \ni t).
$$

- Score each page by total click traffic in/out:

$$
\text{score}(p) = \sum_{i:\, \text{prev}_i = p} n_i \;+\; \sum_{i:\, \text{curr}_i = p} n_i .
$$

- Keep the `max_nodes` most popular pages and restrict edges to transitions between them.
- **Topic used:** `Football` → **small graph** ($n = 200$) and **large graph** ($n = 996$).

### 3. Weighted transition matrix
Accumulate click weights into a weighted adjacency matrix $W \in \mathbb{R}^{n\times n}_{\ge 0}$ with $w_{ij} = \sum n_{ij}$.

### 4. Baseline Markov matrix
Row-normalize $W$ to obtain $P_0$:

$$
P_0(i,j) =
\begin{cases}
\dfrac{w_{ij}}{\sum_j w_{ij}}, & \text{if } \sum_j w_{ij} > 0, \\[2mm]
0, & \text{otherwise.}
\end{cases}
$$

### 5. Teleportation (handling dangling rows)
PageRank-style smoothing guarantees ergodicity and ensures **every row sums to 1**, fixing dangling (zero out-degree) nodes:

$$
\tilde{P}_0 = (1-\alpha)\,P_0 + \alpha\, \mathbf{1}\,v^\top, \qquad v = \tfrac{1}{n}\mathbf{1}, \qquad \alpha = 0.05 .
$$

The baseline stationary distribution $\pi_{\text{base}} = \pi(\tilde{P}_0)$ is then computed for comparison.

### 6. Target distribution
For all experiments the target is **uniform**, $\pi^\* = \tfrac{1}{n}\mathbf{1}$, so the optimization studies how much editing is needed to *flatten* topic-specific popularity.

---

## Mathematical Formulation

To make the $\ell_1$ objective linear, we introduce an auxiliary slack variable $t_{ij} \ge 0$ for each editable entry that upper-bounds the absolute deviation.

**Decision variables**
- $P_{ij}$ — edited transition probabilities.
- $t_{ij}$ — slack variables with $t_{ij} \ge \lvert P_{ij} - \tilde{P}_{ij}\rvert$.
- Non-editable entries are fixed to their baseline: $P_{ij} = \tilde{P}_{ij}$.

**Final LP**

$$
\begin{aligned}
\min_{P,\,t} \quad & \sum_{(i,j)\in\Omega} c_{ij}\, t_{ij} \\
\text{s.t.} \quad
& \sum_{j} P_{ij} = 1, & & \forall i & \text{(row-stochasticity)}\\
& \sum_{i} \pi^\*_i\, P_{ij} = \pi^\*_j, & & \forall j & \text{(stationarity)}\\
& P_{ij} - \tilde{P}_{ij} \le t_{ij}, & & (i,j)\in\Omega & \text{(linearization)}\\
& \tilde{P}_{ij} - P_{ij} \le t_{ij}, & & (i,j)\in\Omega & \text{(linearization)}\\
& P_{ij} \ge 0, \quad t_{ij} \ge 0. & &
\end{aligned}
$$

The stationarity constraint $\pi^{*\top} P = \pi^{*\top}$ is equivalent to the per-column equalities $\sum_i \pi^\*_i P_{ij} = \pi^\*_j$. When $c_{ij} = 1$, the objective is exactly the total entrywise $\ell_1$ change. The LP is **convex, scalable, and sparse** when $\Omega$ is limited to empirical edges.

---

## Feasibility & Challenges

For sparse subgraphs the LP frequently became **infeasible**. The root causes and fixes:

| Cause | Why it breaks | Fix |
|---|---|---|
| **Missing incoming edges** in $\Omega$ | A column $j$ with $\pi^\*_j > 0$ but $\Omega_j = \varnothing$ forces the impossible equality $0 = \pi^\*_j - S_j$ | Add **top-$k$ incoming** edges per node |
| **Row-stochasticity violation** | If most baseline mass lies outside $\Omega$, then $\sum_{j\in\Omega_i} P_{ij} = 1 - B_i$ with $B_i > 1$ is infeasible | Expand $\Omega$ with **top-$k$ outgoing** edges per row |
| **Empty rows after filtering** | A row with no remaining outgoing edge has no editable variable | Always add a **self-loop** $(i,i) \in \Omega$ |
| **Floating-point noise** | Dense constraint matrices caused numerical infeasibility | Store $A_{\text{eq}}, A_{\text{ub}}$ in **CSR sparse** format |

**Key empirical insight — graph density matters.** Sparse topics (`Machine`, `Climate`) stayed infeasible for small $k$, while denser topics (`Football`, `Russia`) became feasible even at $k = 5$. Setting the editable support via the heuristic

$$
\Omega_i \supseteq \{\, j : \tilde{P}_0_{ij} \text{ among the top-}k \,\}, \qquad (i,i)\in\Omega \;\;\forall i,
$$

with $k_{\text{out}} = k_{\text{in}} = 10$ consistently produced feasible problems. Infeasibility is fundamentally a shortage of *degrees of freedom*: when $\Omega$ is too small, $A_{\text{eq}}x = b_{\text{eq}}$ cannot match its right-hand side.

---

## Column Generation (Bonus)

Rather than editing all edges at once, **Column Generation (CG)** maintains a small restricted editable set $\Omega_R$ and grows it using **dual information**.

### Dual problem
With multipliers $\alpha_i$ (row-sum), $\beta_j$ (stationarity), and $u_{ij}, v_{ij} \ge 0$ for the two $\ell_1$ inequalities (set $\gamma_{ij} = u_{ij} - v_{ij}$, so $\lvert\gamma_{ij}\rvert \le c_{ij}$), the dual reads:

$$
\begin{aligned}
\max_{\alpha,\beta,\gamma} \quad & \sum_{i=1}^{n}\alpha_i + \sum_{j=1}^{n}\beta_j\,\pi^\*_j + \sum_{(i,j)\in\Omega}\gamma_{ij}\,\tilde{P}_{ij} \\
\text{s.t.} \quad & \alpha_i + \pi^\*_i\,\beta_j + \gamma_{ij} \le 0, \quad (i,j)\in\Omega, \\
& \lvert \gamma_{ij}\rvert \le c_{ij}, \quad (i,j)\in\Omega .
\end{aligned}
$$

### Pricing via "safe score"
Optimizing $\gamma_{ij}$ over $[-c_{ij}, c_{ij}]$ gives the **reduced-cost score** of a non-basic edge:

$$
s_{ij} = \alpha_i + \pi^\*_i\,\beta_j + c_{ij}.
$$

If $s_{ij} > 0$ for an edge $(i,j) \notin \Omega_R$, adding it can decrease the primal objective.

### CG loop
1. Solve the **Restricted Master Problem (RMP)** on the current $\Omega_R$; obtain primal $(P, t)$ and duals $(\alpha, \beta)$.
2. Compute $s_{ij}$ for all $(i,j) \notin \Omega_R$.
3. Add up to $M$ edges with $s_{ij} > \text{tol}$.
4. Stop when no edge exceeds the tolerance.

The initial $\Omega_R^{(0)}$ includes all self-loops, the top-$k_{\text{out}}$ outgoing edges per row, and the top-$k_{\text{in}}$ incoming edges per column.

---

## Results

### Small Football subgraph ($n = 200$)
LP with $\lvert\Omega_R\rvert = 4{,}323$ editable edges:

| Metric | Value |
|---|---|
| Edit cost $\|\Delta\|_1$ | **106.3** |
| $\|\Delta\|_\infty$ | $\approx 0.95$ |
| Sparsity | $\approx 0.97$ (only ~3% of entries change) |
| Stationary deviation $\|\pi(P_{\text{opt}}) - \pi^\*\|_1$ | $\approx 10^{-15}$ |

The skewed baseline exposure is **strongly flattened** to match the uniform target, with edits concentrated on a sparse band of high-traffic hub pages. A mixing proxy $\|P^k - \mathbf{1}\pi^{*\top}\|_1$ decays from ~350 ($k=1$) to ~60 ($k=20$).

### Large Football subgraph ($n = 996$)
LP with $\lvert\Omega_R\rvert \approx 20{,}000$ editable edges — the solver stays numerically stable and converges quickly:

| Metric | Value |
|---|---|
| Edit cost $\|\Delta\|_1$ | $\approx 565.4$ |
| $\|\Delta\|_\infty$ | $\approx 0.95$ |
| Sparsity | $\approx 0.99$ (<1% of entries change) |
| Stationary deviation | machine precision |

### Ablations
- **$\ell_1$ vs $\ell_\infty$ objective:** $\ell_1$ yields sparse edits at low cost ($\approx 106$ for $n=200$); $\ell_\infty$ spreads edits and raises cost to $\approx 349$.
- **Uniform vs degree-weighted costs** ($c_{ij} = 1/\max(\text{outdeg}(i), \varepsilon)$): edit patterns become more balanced; total cost rises slightly ($106 \to 111$); sparsity stays high.
- **Edge-set size:** too-small $k$ → infeasible; $k_{\text{out}} = k_{\text{in}} = 10$ reliably feasible.
- **Perturbation robustness:** for $\pi^\*_{\text{pert}} = \pi^\* + \varepsilon\eta$ with $\varepsilon \in \{10^{-6}, 10^{-5}, 2\cdot 10^{-5}\}$, feasibility holds, cost scales linearly with $\varepsilon$, and sparsity stays above 0.88.

### Column Generation (20-node subgraph)
| Quantity | Value |
|---|---|
| Initial editable set $\lvert\Omega_R^{(0)}\rvert$ | 95 |
| Initial objective | 8.190626 |
| Final editable set $\lvert\Omega_R\rvert$ | 115 |
| **Final CG objective** | **8.072638** |
| Full LP (fixed support) objective | 8.190626 |

CG automatically discovers worthwhile edges, lowering the objective from **8.19 → 8.07** while satisfying all constraints and avoiding the full dense $20\times20$ LP — a promising route to scaling TSDP to larger graphs.

---


## Visualizations

The project generates two main figure types per subgraph:

- **Exposure distribution comparison** — a grouped bar chart of the top-20 pages showing the baseline $\pi(\tilde{P}_0)$, the uniform target $\pi^\*$, and the optimized $\pi(P_{\text{opt}})$. It visually confirms how the skewed baseline collapses onto the flat target.
- **Edit heatmap** — $\Delta = \lvert P_{\text{opt}} - \tilde{P}_0\rvert$ rendered as a heatmap, revealing that edits stay extremely sparse and cluster around highly connected hub nodes.

---


## Authors

- **Akhmetova Albina**
- **Naumov Dmitry**

*November 2025*

---

<p align="center"><i>If you find this project useful, consider giving it a ⭐!</i></p>
