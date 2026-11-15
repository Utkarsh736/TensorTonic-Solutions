import numpy as np

def initialize_mcts_edges(policy_logits: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Returns: N as an int64 array; W, Q, and P as float64 arrays in a dictionary.
    """
    logits = np.asarray(policy_logits)
    mask = np.asarray(legal_mask)

    n_actions = logits.shape[0]

    legal_indices = np.flatnonzero(mask)

    legal_logits = logits[legal_indices]

    m = np.max(legal_logits)
    exp_logits = np.exp(legal_logits - m)
    sum_exp = np.sum(exp_logits)
    legal_probs = exp_logits / sum_exp

    P = np.zeros(n_actions, dtype=np.float64)
    P[legal_indices] = legal_probs

    N = np.zeros(n_actions, dtype=np.int64)
    W = np.zeros(n_actions, dtype=np.float64)
    Q = np.zeros(n_actions, dtype=np.float64)

    return {"N": N, "W": W, "Q": Q, "P": P}
