import random
import numpy as np

from simulations import Simulation
from population import create_population

# ── Reproducibility ───────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

# ── Population ────────────────────────────────────────────────────
agents = create_population(
    n_cooperators=0,
    n_defectors=0,
    n_tft=0,
    n_unforgiving=10,
    n_pheromones=0,
    n_learners=10,
    learner_params=learner_params,
    n_learners_ext=0,
    learner_ext_params=learner_ext_params,
)

# ── Simulation ────────────────────────────────────────────────────
sim = Simulation(
    agents=agents,
    ticks_per_episode=200,
    episodes=10000
)

sim.run(print_every=500)

sim.plot(
    smooth_window=100,
    save_path="results.png",
)
