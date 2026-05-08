import random
import numpy as np

from simulation2d import Simulation2D
from population import create_population

# ── Reproducibility ───────────────────────────────────────────────
SEED = 42

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
    episodes=1000,
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

random.seed(SEED)
np.random.seed(SEED)

agents_on = create_population(**POPULATION_KWARGS)

sim_on = Simulation2D(
    agents=agents_on,
    chemotaxis=True,
    **SIM_KWARGS,
)

sim_on.run(print_every=1, live=True, live_fps=4, live_skip=1)
sim_on.plot(smooth_window=50, save_path="results2d.png")
