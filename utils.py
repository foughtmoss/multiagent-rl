
import random


def e_greedy_policy(e, qtable, state):
    """
    Implements the epsilon-greedy policy.
    qtable[state] is a list of Q-values, index = action
    """

    # Converti lo stato in tuple (chiave valida)
    state_key = tuple(state)
    q_values = qtable[state_key]
    actions = list(range(len(q_values)))  # [0, 1]

    # Esplorazione
    if random.random() < e:
        return random.choice(actions)

    # Sfruttamento
    max_q = max(q_values)
    best_actions = [a for a in actions if q_values[a] == max_q]

    return random.choice(best_actions)


def compute_payoff(agent_action, partner_action):
    if agent_action == 0 and partner_action == 0:  # C-C
        return 3
    elif agent_action == 0 and partner_action == 1:  # C-D
        return 0
    elif agent_action == 1 and partner_action == 0:  # D-C
        return 5
    else:  # D-D
        return 1


def get_reward(agent_action, partner_action, base_reward=9):
    agent_payoff = compute_payoff(agent_action, partner_action)
    partner_payoff = compute_payoff(partner_action, agent_action)

    if agent_payoff > partner_payoff:
        reward = base_reward
    elif agent_payoff < partner_payoff:
        reward = -base_reward
    else:
        reward = base_reward / 2

    return reward