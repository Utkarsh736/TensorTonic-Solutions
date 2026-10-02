import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Returns diversity_score and is_collapsed in a dictionary.
    """
    diversity_score = float(np.mean(np.std(generated_samples, axis=0)))

    return {
        "diversity_score": diversity_score,
        "is_collapsed": (diversity_score<threshold),
    }