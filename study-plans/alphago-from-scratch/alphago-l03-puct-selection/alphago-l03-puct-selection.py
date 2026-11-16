import numpy as np

def select_puct_action(visits: np.ndarray, total_values: np.ndarray, priors: np.ndarray, legal_mask: np.ndarray, cpuct: float) -> tuple[int, np.ndarray]:
    """
    Returns: a tuple of the Python action index and the float64 score array.
    """
    visits_f = np.asarray(visits, dtype=np.float64).copy()
    values_f = np.asarray(total_values, dtype=np.float64).copy()
    priors_f = np.asarray(priors, dtype=np.float64).copy()
    legal = np.asarray(legal_mask, dtype=bool).copy()

    n = visits_f.shape[0]

    q = np.zeros(n, dtype=np.float64)
    np.divide(values_f, visits_f, out=q, where=visits_f > 0)

    total_visits = float(np.sum(visits_f))

    bonus = cpuct * priors_f * np.sqrt(total_visits) / (1.0 + visits_f)

    scores = q + bonus

    scores[~legal] = -np.inf

    selected_action = int(np.argmax(scores))

    return selected_action, scores
