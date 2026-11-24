import numpy as np

def neural_guided_mcts(
    root_state: str,
    transitions: dict,
    legal_actions: dict,
    players: dict,
    policy_logits: dict,
    value_predictions: dict,
    terminal_values: dict,
    simulations: int,
    cpuct: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns root visit counts, mean values, and policy arrays in legal-action order.
    """
    cpuct = float(cpuct)
    temperature = float(temperature)

    stats = {}

    def expand(state):
        """Expand a nonterminal state, creating its edge statistics."""
        actions = legal_actions[state]
        logits = np.asarray(policy_logits[state], dtype=np.float64)

        # Stable softmax over the supplied logits.
        m = np.max(logits)
        exp_logits = np.exp(logits - m)
        priors = exp_logits / np.sum(exp_logits)

        a = len(actions)
        stats[state] = {
            "N": np.zeros(a, dtype=np.int64),
            "W": np.zeros(a, dtype=np.float64),
            "Q": np.zeros(a, dtype=np.float64),
            "P": priors.astype(np.float64),
        }

    def child_state(state, action):
        """Look up a child, supporting both int and string action keys."""
        children = transitions[state]
        if action in children:
            return children[action]
        return children[str(action)]

    # The root starts expanded.
    expand(root_state)

    for _ in range(simulations):
        path = []  # (parent_state, action_index, child_state)
        state = root_state

        while state in stats:
            st = stats[state]
            N = st["N"]
            Q = st["Q"]
            P = st["P"]

            total_visits = float(np.sum(N))
            bonus = (
                cpuct
                * P
                * np.sqrt(total_visits)
                / (1.0 + N.astype(np.float64))
            )
            scores = Q + bonus

            idx = int(np.argmax(scores))
            action = legal_actions[state][idx]
            nxt = child_state(state, action)

            path.append((state, idx, nxt))
            state = nxt

        if state in terminal_values:
            leaf_value = float(terminal_values[state])
            is_terminal = True
        else:
            leaf_value = float(value_predictions[state])
            is_terminal = False

        leaf_player = players[state]

        if not is_terminal:
            expand(state)

        for parent, idx, _ in reversed(path):
            parent_player = players[parent]

            if parent_player == leaf_player:
                signed_value = leaf_value
            else:
                signed_value = -leaf_value

            st = stats[parent]
            st["N"][idx] += 1
            st["W"][idx] += signed_value
            st["Q"][idx] = st["W"][idx] / st["N"][idx]

    root_stats = stats[root_state]
    visits = root_stats["N"].astype(np.int64).copy()
    mean_values = root_stats["Q"].astype(np.float64).copy()

    if temperature == 0.0:
        policy = np.zeros_like(visits, dtype=np.float64)
        best_idx = int(np.argmax(visits))  # first max = lowest action
        policy[best_idx] = 1.0
    else:
        n_float = visits.astype(np.float64)

        if np.all(n_float == 0):
            # No visits → uniform legal policy.
            policy = np.full(
                visits.shape[0], 1.0 / visits.shape[0], dtype=np.float64
            )
        else:
            # Log weights for numerical stability at small temperatures.
            log_weights = np.full_like(n_float, -np.inf, dtype=np.float64)
            positive = n_float > 0
            log_weights[positive] = np.log(n_float[positive]) / temperature

            max_log = np.max(log_weights)
            weights = np.exp(log_weights - max_log)
            policy = weights / np.sum(weights)

    return visits, mean_values, policy
