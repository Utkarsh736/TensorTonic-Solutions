import numpy as np

def visit_count_policy(
    visits: np.ndarray,
    legal_mask: np.ndarray,
    temperature: float,
    uniform_draw: float | None,
) -> tuple[np.ndarray, int]:
    """
    Returns the float64 policy array and selected action index.
    """
    visits_arr = np.asarray(visits)
    legal = np.asarray(legal_mask, dtype=bool)

    n = visits_arr.shape[0]
    legal_indices = np.flatnonzero(legal)

    policy = np.zeros(n, dtype=np.float64)

    if temperature == 0:
        legal_visits = visits_arr[legal_indices]
        max_count = np.max(legal_visits)
        best_candidates = legal_indices[legal_visits == max_count]
        selected_action = int(best_candidates[0])
        policy[selected_action] = 1.0
        return policy, selected_action

    legal_visits = visits_arr[legal_indices].astype(np.float64)

    if np.all(legal_visits == 0):
        probs_legal = np.full(legal_indices.size, 1.0 / legal_indices.size,
                              dtype=np.float64)
    else:
        log_weights = np.full(legal_indices.size, -np.inf, dtype=np.float64)
        positive = legal_visits > 0
        log_weights[positive] = np.log(legal_visits[positive]) / temperature

        max_log = np.max(log_weights)
        weights = np.exp(log_weights - max_log)
        total = np.sum(weights)
        probs_legal = weights / total

    policy[legal_indices] = probs_legal

    cumulative = np.cumsum(probs_legal)
    idx = int(np.searchsorted(cumulative, uniform_draw, side="right"))

    if idx >= legal_indices.size:
        idx = legal_indices.size - 1

    selected_action = int(legal_indices[idx])
    return policy, selected_action
