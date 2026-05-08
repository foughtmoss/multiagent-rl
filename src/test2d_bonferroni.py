import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu

from simulation2d import Simulation2D
from population import create_population

# ── Hyperparameters ───────────────────────────────────────────────
BASE_PARAMS = dict(
    lr=0.1,
    gamma=0.9,
    eps_start=1.0,
    eps_min=0.01,
    eps_decay=0.999675,
)

learner_params = dict(**BASE_PARAMS, max_obs_hist=1)
learner_ext_params = dict(**BASE_PARAMS, max_obs_hist=3)

SIM_KWARGS = dict(
    grid_rows=30,
    grid_cols=30,
    ticks_per_episode=200,
    episodes=3000,
    diffusion=0.9,
    evaporation=0.9,
    phero_drop=3.0,
    sniff_threshold=1.0,
)

POPULATION_KWARGS = dict(
    n_cooperators=10,
    n_defectors=10,
    n_tft=10,
    n_unforgiving=10,
    n_pheromones=10,
    n_learners=10,
    learner_params=learner_params,
    n_learners_ext=10,
    learner_ext_params=learner_ext_params,
)

# ── ANALYSIS SETTINGS ─────────────────────────────────────────────
N_RUNS = 2
WARMUP = 1000

STRATEGIES = [
    "Cooperator", "Defector", "TitForTat", "Unforgiving",
    "Pheromones", "Learner", "LearnerExt",
]

results = {s: [] for s in STRATEGIES}

# ── RUN MULTIPLI ──────────────────────────────────────────────────
for seed in range(N_RUNS):
    print(f"\n=== RUN {seed} ===")

    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(**POPULATION_KWARGS)

    sim = Simulation2D(
        agents=agents,
        chemotaxis=True,
        **SIM_KWARGS,
    )

    sim.run(print_every=2000, live=False)

    for strategy in results:
        scores = sim.ep_avg_score[strategy]
        final_score = np.mean(scores[WARMUP:])
        results[strategy].append(final_score)

# ── BOOTSTRAP CI ──────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=10000, ci=95):
    rng = np.random.default_rng()
    means = [np.mean(rng.choice(data, len(data), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi

# ── STATISTICHE DESCRITTIVE ───────────────────────────────────────
print("\n===== DESCRIPTIVE STATISTICS =====")
summary = {}
for strategy in STRATEGIES:
    vals = np.array(results[strategy])
    mean = np.mean(vals)
    std  = np.std(vals, ddof=1)
    lo, hi = bootstrap_ci(vals)
    summary[strategy] = dict(mean=mean, std=std, ci_lo=lo, ci_hi=hi)
    print(f"{strategy:12s}  mean={mean:7.1f}  std={std:6.2f}"
          f"  Bootstrap CI95=[{lo:.1f}, {hi:.1f}]")

# ── KRUSKAL-WALLIS ────────────────────────────────────────────────
all_vals = [np.array(results[s]) for s in STRATEGIES]
stat_kw, p_kw = kruskal(*all_vals)
print(f"\nKruskal-Wallis: H={stat_kw:.3f}, p={p_kw:.6f}")
if p_kw < 0.05:
    print("→ Differenze significative tra le strategie (p < 0.05). Procedo con post-hoc.")
else:
    print("→ Nessuna differenza significativa globale.")

# ── MANN-WHITNEY + BONFERRONI ─────────────────────────────────────
pairs = list(combinations(STRATEGIES, 2))
n_comparisons = len(pairs)
alpha = 0.05
alpha_bonf = alpha / n_comparisons

print(f"\nMann-Whitney pairwise (Bonferroni α={alpha_bonf:.5f}, {n_comparisons} confronti):")
pairwise = {}
for s1, s2 in pairs:
    u, p = mannwhitneyu(results[s1], results[s2], alternative='two-sided')
    pairwise[(s1, s2)] = p
    sig = "***" if p < alpha_bonf else ("*  " if p < 0.05 else "   ")
    print(f"  {s1:12s} vs {s2:12s}  p={p:.5f} {sig}")

# ── SALVA CSV ─────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("\nSalvato results.csv")

# ── PLOT 1: BOXPLOT ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(
    [results[s] for s in STRATEGIES],
    labels=STRATEGIES,
    patch_artist=True,
    medianprops=dict(color="orange", linewidth=2),
)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b",
          "#e377c2", "#d62728", "#9467bd"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# annota media con bootstrap CI
for i, s in enumerate(STRATEGIES, start=1):
    lo = summary[s]["ci_lo"]
    hi = summary[s]["ci_hi"]
    ax.vlines(i, lo, hi, color="black", linewidth=2.5, zorder=5)

ax.set_ylabel("Average score (post-warmup)")
ax.set_title("Performance distribution across runs\n"
             "(barre nere = Bootstrap CI 95%)")
ax.set_xticklabels(STRATEGIES, rotation=30, ha="right")
plt.tight_layout()
plt.savefig("boxplot.png", dpi=150)
plt.show()
print("Salvato boxplot.png")

# ── PLOT 2: HEATMAP P-VALUE ───────────────────────────────────────
n = len(STRATEGIES)
p_matrix = np.ones((n, n))

for (s1, s2), p in pairwise.items():
    i = STRATEGIES.index(s1)
    j = STRATEGIES.index(s2)
    p_matrix[i, j] = p
    p_matrix[j, i] = p  # simmetrica

# annota con stelle
def p_to_annot(p):
    if p < alpha_bonf:
        return f"{p:.4f}\n***"
    elif p < 0.05:
        return f"{p:.4f}\n*"
    else:
        return f"{p:.4f}"

annot_matrix = np.empty((n, n), dtype=object)
for i in range(n):
    for j in range(n):
        if i == j:
            annot_matrix[i, j] = "—"
        else:
            annot_matrix[i, j] = p_to_annot(p_matrix[i, j])

# maschera diagonale per colori (diagonale a 1 confonde la colormap)
mask_diag = np.eye(n, dtype=bool)
display_matrix = p_matrix.copy()
display_matrix[mask_diag] = np.nan

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    display_matrix,
    annot=annot_matrix,
    fmt="",
    xticklabels=STRATEGIES,
    yticklabels=STRATEGIES,
    cmap="RdYlGn",
    vmin=0,
    vmax=0.05,
    linewidths=0.5,
    linecolor="gray",
    ax=ax,
    cbar_kws={"label": "p-value"},
    mask=mask_diag,
)
ax.set_title(
    f"Mann-Whitney p-values (pairwise)\n"
    f"*** = significativo dopo Bonferroni (α={alpha_bonf:.5f})  "
    f"* = p < 0.05 non corretto\n"
    f"Rosso = differenza significativa, Verde = non significativa"
)
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("pvalue_heatmap.png", dpi=150)
plt.show()
print("Salvato pvalue_heatmap.png")

# ── PLOT 3: MEAN + CI BARPLOT ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
means = [summary[s]["mean"] for s in STRATEGIES]
ci_lo = [summary[s]["mean"] - summary[s]["ci_lo"] for s in STRATEGIES]
ci_hi = [summary[s]["ci_hi"] - summary[s]["mean"] for s in STRATEGIES]

x = np.arange(len(STRATEGIES))
bars = ax.bar(x, means, color=colors, alpha=0.75, edgecolor="black", linewidth=0.7)
ax.errorbar(x, means, yerr=[ci_lo, ci_hi],
            fmt="none", color="black", capsize=5, linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels(STRATEGIES, rotation=30, ha="right")
ax.set_ylabel("Average score (post-warmup)")
ax.set_title("Mean score per strategy ± Bootstrap CI 95%")
plt.tight_layout()
plt.savefig("barplot_ci.png", dpi=150)
plt.show()
print("Salvato barplot_ci.png")