import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu

# ── LOAD CSV ───────────────────────────────────────────────────────
df = pd.read_csv("results/results_10_run.csv")
print(f"Loaded {len(df)} runs")
print(f"Strategies: {df.columns.tolist()}\n")

# ── RECREATE DICT results ──────────────────────────────────────────
results = {col: df[col].values for col in df.columns}
STRATEGIES = list(results.keys())

# ── BOOTSTRAP CI ──────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=10000, ci=95):
    rng = np.random.default_rng()
    means = [np.mean(rng.choice(data, len(data), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi

# ── DESCRIPTIVE STATISTICS ────────────────────────────────────────
print("===== DESCRIPTIVE STATISTICS =====")
summary = {}
for strategy in STRATEGIES:
    vals = np.array(results[strategy])
    mean = np.mean(vals)
    std  = np.std(vals, ddof=1)
    lo, hi = bootstrap_ci(vals)
    summary[strategy] = dict(mean=mean, std=std, ci_lo=lo, ci_hi=hi)
    print(f"{strategy:12s}  mean={mean:7.1f}  std={std:6.2f}"
          f"  Bootstrap CI95=[{lo:.1f}, {hi:.1f}]")

# ── KRUSKAL-WALLIS ─────────────────────────────────────────────────
all_vals = [np.array(results[s]) for s in STRATEGIES]
stat_kw, p_kw = kruskal(*all_vals)
print(f"\nKruskal-Wallis: H={stat_kw:.3f}, p={p_kw:.6f}")
if p_kw < 0.05:
    print("→ Significant differences between strategies (p < 0.05). Proceeding with post-hoc tests.")
else:
    print("→ No significant global difference.")

# ── MANN-WHITNEY + BONFERRONI ──────────────────────────────────────
pairs = list(combinations(STRATEGIES, 2))
n_comparisons = len(pairs)
alpha = 0.05
alpha_bonf = alpha / n_comparisons

print(f"\nMann-Whitney pairwise (Bonferroni α={alpha_bonf:.5f}, {n_comparisons} comparisons):")
pairwise = {}
for s1, s2 in pairs:
    u, p = mannwhitneyu(results[s1], results[s2], alternative='two-sided')
    pairwise[(s1, s2)] = p
    sig = "***" if p < alpha_bonf else ("*  " if p < 0.05 else "   ")
    print(f"  {s1:12s} vs {s2:12s}  p={p:.5f} {sig}")

# ── PLOT 1: BOXPLOT ────────────────────────────────────────────────
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

# annotate mean with bootstrap CI
for i, s in enumerate(STRATEGIES, start=1):
    lo = summary[s]["ci_lo"]
    hi = summary[s]["ci_hi"]
    ax.vlines(i, lo, hi, color="black", linewidth=2.5, zorder=5)

ax.set_ylabel("Average score (post-warmup)")
ax.set_title("Performance distribution across runs\n"
             "(black bars = Bootstrap CI 95%)")
ax.set_xticklabels(STRATEGIES, rotation=30, ha="right")
plt.tight_layout()
plt.savefig("boxplot.png", dpi=150)
plt.show()
print("Saved boxplot.png")

# ── PLOT 2: P-VALUE HEATMAP ────────────────────────────────────────
n = len(STRATEGIES)
p_matrix = np.ones((n, n))

for (s1, s2), p in pairwise.items():
    i = STRATEGIES.index(s1)
    j = STRATEGIES.index(s2)
    p_matrix[i, j] = p
    p_matrix[j, i] = p  # symmetric

# annotate with significance stars
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

# mask diagonal (diagonal at 1 distorts the colormap)
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
    f"*** = significant after Bonferroni (α={alpha_bonf:.5f})  "
    f"* = p < 0.05 uncorrected\n"
    f"Red = significant difference, Green = not significant"
)
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("pvalue_heatmap.png", dpi=150)
plt.show()
print("Saved pvalue_heatmap.png")

# ── PLOT 3: MEAN + CI BARPLOT ──────────────────────────────────────
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
print("Saved barplot_ci.png")

# ── SAVE SUMMARY TO CSV ────────────────────────────────────────────
summary_df = pd.DataFrame(summary).T
summary_df.to_csv("summary_statistics.csv")
print("\nSaved summary_statistics.csv")

# ── SAVE P-VALUES TO CSV ───────────────────────────────────────────
p_df = pd.DataFrame(p_matrix, index=STRATEGIES, columns=STRATEGIES)
p_df.to_csv("pairwise_pvalues.csv")
print("Saved pairwise_pvalues.csv")