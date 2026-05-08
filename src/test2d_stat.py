import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

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
    episodes=20000,
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
N_RUNS = 10
WARMUP = 8000

results = {
    "Cooperator": [],
    "Defector": [],
    "TitForTat": [],
    "Unforgiving": [],
    "Pheromones": [],
    "Learner": [],
    "LearnerExt": [],
}

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

    # estrazione risultati
    for strategy in results:
        scores = sim.ep_avg_score[strategy]
        final_score = np.mean(scores[WARMUP:])
        results[strategy].append(final_score)

# ── STATISTICA ────────────────────────────────────────────────────
print("\n===== FINAL STATISTICS =====")

summary = {}

for strategy, vals in results.items():
    vals = np.array(vals)
    n = len(vals)

    mean = np.mean(vals)
    std = np.std(vals, ddof=1)

    t_value = t.ppf(0.975, df=n - 1)
    ci = t_value * std / np.sqrt(n)

    summary[strategy] = (mean, std, ci)

    print(f"{strategy:12s}  mean={mean:7.1f}  std={std:6.2f}  CI95=±{ci:.2f}")

# ── SALVA RISULTATI ───────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\nSaved results to results.csv")

# ── BOXPLOT ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.boxplot([df[col] for col in df.columns], labels=df.columns)
plt.xticks(rotation=30)
plt.ylabel("Average score")
plt.title("Performance distribution across runs")
plt.tight_layout()
plt.savefig("boxplot.png", dpi=150)
plt.show()

print("Saved boxplot to boxplot.png")

