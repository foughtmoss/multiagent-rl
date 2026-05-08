from players import Cooperator, Defector, TitForTat, Learner, Learner_Extended_History, Unforgiving, Pheromones


def create_population(n_cooperators=0, n_defectors=0, n_tft=0,
                      n_unforgiving=0, n_pheromones=0,
                      n_learners=0, learner_params=None,
                      n_learners_ext=0, learner_ext_params=None):
    """
    Returns a flat list of Agent instances with sequential integer ids.

    n_learners      -> standard Learner (1-step state)
    n_learners_ext  -> Learner_Extended_History (max_obs_hist-step state)
    n_unforgiving   -> Unforgiving (Grim Trigger)
    n_pheromones    -> Pheromones (TitForTat + cooperation signal)
    """
    if learner_params is None:
        learner_params = {}
    if learner_ext_params is None:
        learner_ext_params = {}

    # Reset shared pheromone signal when building a new population
    Pheromones.reset_signal()

    agents = []
    next_id = 0

    for _ in range(n_cooperators):
        agents.append(Cooperator(id=next_id)); next_id += 1
    for _ in range(n_defectors):
        agents.append(Defector(id=next_id)); next_id += 1
    for _ in range(n_tft):
        agents.append(TitForTat(id=next_id)); next_id += 1
    for _ in range(n_unforgiving):
        agents.append(Unforgiving(id=next_id)); next_id += 1
    for _ in range(n_pheromones):
        agents.append(Pheromones(id=next_id)); next_id += 1
    for _ in range(n_learners):
        agents.append(Learner(id=next_id, **learner_params)); next_id += 1
    for _ in range(n_learners_ext):
        agents.append(Learner_Extended_History(id=next_id, **learner_ext_params)); next_id += 1

    return agents