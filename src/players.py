from abc import ABC, abstractmethod
from collections import deque
from utils import compute_payoff, e_greedy_policy


class Agent(ABC):
    def __init__(self, id):
        self.id = id
        self.partnered = False
        self.partner = None
        self.last_action = None
        self.score = 0

    @abstractmethod
    def choose_action(self):
        pass

    def play(self, partner):
        self.partnered = True
        self.partner = partner
        self.last_action = self.choose_action()
        return self.last_action

    def stop(self):
        self.partner = None
        self.partnered = False

    def observe(self, partner_action):
        pass

    def update_score(self):
        self.score += compute_payoff(agent_action=self.last_action,
                                     partner_action=self.partner.last_action)
        return self.score


class Cooperator(Agent):
    def __init__(self, id):
        super().__init__(id)

    def choose_action(self):
        return 0


class Defector(Agent):
    def __init__(self, id):
        super().__init__(id)

    def choose_action(self):
        return 1


class TitForTat(Agent):
    def __init__(self, id):
        super().__init__(id)
        self.partner_history = {}

    def choose_action(self):
        if self.partner.id not in self.partner_history:
            self.partner_history[self.partner.id] = None
            return 0
        pa = self.partner_history[self.partner.id]
        return 0 if pa == 0 else 1

    def observe(self, partner_action):
        self.partner_history[self.partner.id] = partner_action


class Learner(Agent):
    def __init__(self, id, lr, gamma, max_obs_hist, eps_start, eps_min, eps_decay):
        """
        Per-partner Q-tables.

        State = (own_last_action, partner_last_action) with sentinel -1.
        One independent Q-table per partner.
        """
        super().__init__(id)
        self.lr = lr
        self.gamma = gamma
        self.max_obs_hist = max_obs_hist
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.epsilon = eps_start

        self.actions = [0, 1]

        self.qtables = {}
        self.own_last = {}
        self.partner_last = {}

        self.prev_pid = None
        self.prev_state = None
        self.prev_action = None

    def _ensure_partner(self, pid):
        if pid not in self.qtables:
            self.qtables[pid] = {}

    def _get_state(self, pid):
        return (self.own_last.get(pid, -1), self.partner_last.get(pid, -1))

    def _ensure_state(self, pid, state):
        if state not in self.qtables[pid]:
            self.qtables[pid][state] = [0.0, 0.0]

    def choose_action(self):
        pid = self.partner.id
        self._ensure_partner(pid)
        state = self._get_state(pid)
        self._ensure_state(pid, state)

        action = e_greedy_policy(self.epsilon, self.qtables[pid], state)

        self.prev_pid = pid
        self.prev_state = state
        self.prev_action = action
        return action

    def observe(self, partner_action):
        pid = self.partner.id
        self._ensure_partner(pid)
        self.partner_last[pid] = partner_action
        self.own_last[pid] = self.last_action

    def learn(self, reward):
        if self.prev_state is None:
            return
        pid = self.prev_pid
        next_state = self._get_state(pid)

        self._ensure_state(pid, self.prev_state)
        self._ensure_state(pid, next_state)

        qt = self.qtables[pid]
        td = (reward + self.gamma * max(qt[next_state])
              - qt[self.prev_state][self.prev_action])
        qt[self.prev_state][self.prev_action] += self.lr * td

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    @property
    def qtable(self):
        merged = {}
        for pid, qt in self.qtables.items():
            for state, vals in qt.items():
                if state not in merged or max(vals) > max(merged[state]):
                    merged[state] = vals
        return merged


class Learner_Extended_History(Agent):
    def __init__(self, id, lr, gamma, max_obs_hist, eps_start, eps_min, eps_decay):
        """
        Per-partner Q-tables with extended history.

        State = (own_t-k, ..., own_t-1, partner_t-k, ..., partner_t-1)
        length = 2 * max_obs_hist, sentinel -1 for missing observations.

        With max_obs_hist=1 this reduces to the standard Learner.
        WARNING: state space grows as 3^(2*max_obs_hist) — keep max_obs_hist <= 5.
        """
        super().__init__(id)
        self.lr = lr
        self.gamma = gamma
        self.max_obs_hist = max_obs_hist
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.epsilon = eps_start

        self.actions = [0, 1]

        self.qtables = {}  # qtables[pid]      -> {state_tuple: [Q_C, Q_D]}
        self.own_hist = {}  # own_hist[pid]     -> deque of last k own actions
        self.partner_hist = {}  # partner_hist[pid] -> deque of last k partner actions

        self.prev_pid = None
        self.prev_state = None
        self.prev_action = None

    def _ensure_partner(self, pid):
        if pid not in self.qtables:
            self.qtables[pid] = {}
            self.own_hist[pid] = deque([-1] * self.max_obs_hist,
                                       maxlen=self.max_obs_hist)
            self.partner_hist[pid] = deque([-1] * self.max_obs_hist,
                                           maxlen=self.max_obs_hist)

    def _get_state(self, pid):
        return tuple(self.own_hist[pid]) + tuple(self.partner_hist[pid])

    def _ensure_state(self, pid, state):
        if state not in self.qtables[pid]:
            self.qtables[pid][state] = [0.0, 0.0]

    def choose_action(self):
        pid = self.partner.id
        self._ensure_partner(pid)
        state = self._get_state(pid)
        self._ensure_state(pid, state)

        action = e_greedy_policy(self.epsilon, self.qtables[pid], state)

        self.prev_pid = pid
        self.prev_state = state
        self.prev_action = action
        return action

    def observe(self, partner_action):
        pid = self.partner.id
        self._ensure_partner(pid)
        self.own_hist[pid].append(self.last_action)
        self.partner_hist[pid].append(partner_action)

    def learn(self, reward):
        if self.prev_state is None:
            return
        pid = self.prev_pid
        next_state = self._get_state(pid)

        self._ensure_state(pid, self.prev_state)
        self._ensure_state(pid, next_state)

        qt = self.qtables[pid]
        td = (reward + self.gamma * max(qt[next_state])
              - qt[self.prev_state][self.prev_action])
        qt[self.prev_state][self.prev_action] += self.lr * td

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    @property
    def qtable(self):
        merged = {}
        for pid, qt in self.qtables.items():
            for state, vals in qt.items():
                if state not in merged or max(vals) > max(merged[state]):
                    merged[state] = vals
        return merged


class Unforgiving(Agent):
    """
    Cooperates until the partner defects once, then defects forever
    with that partner (no forgiveness).
    """

    def __init__(self, id):
        super().__init__(id)
        self.betrayed_by = set()  # partner ids that have defected at least once

    def choose_action(self):
        if self.partner.id in self.betrayed_by:
            return 1  # defect forever
        return 0  # cooperate until betrayed

    def observe(self, partner_action):
        if partner_action == 1:
            self.betrayed_by.add(self.partner.id)


class Pheromones(Agent):
    """
    Behaves like TitForTat by default.
    If the shared cooperation signal exceeds a threshold,
    cooperates unconditionally (trusting environment).
    Emits signal when cooperating; signal evaporates each tick.

    Parameters (class-level, shared among all Pheromones agents):
        _evaporation : multiplicative decay per tick (default 0.9)
        _drop        : signal emitted when cooperating (default 3.0)
        _threshold   : signal level above which unconditional coop kicks in (default 1.0)
    """
    _signal = 0.0
    _evaporation = 0.9
    _drop = 3.0
    _threshold = 1.0

    def __init__(self, id):
        super().__init__(id)
        self.partner_history = {}

    # -- class-level signal management --------------------------------
    @classmethod
    def evaporate(cls):
        cls._signal *= cls._evaporation

    @classmethod
    def emit(cls):
        cls._signal += cls._drop

    @classmethod
    def reset_signal(cls):
        cls._signal = 0.0

    # -- per-agent logic ----------------------------------------------
    def choose_action(self):
        if Pheromones._signal >= Pheromones._threshold:
            return 0  # cooperate unconditionally — environment is cooperative
        # fallback: TitForTat
        if self.partner.id not in self.partner_history:
            return 0
        return self.partner_history[self.partner.id]

    def observe(self, partner_action):
        self.partner_history[self.partner.id] = partner_action
        if self.last_action == 0:  # I cooperated → emit signal
            Pheromones.emit()
