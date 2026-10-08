import numpy as np

def apply_mlm_mask(token_ids: np.ndarray, mask_positions: np.ndarray,
                   replace_probs: np.ndarray, random_tokens: np.ndarray,
                   mask_token_id: int = 103) -> dict:
    """
    Returns masked_ids and labels as int64 arrays in a dictionary.
    """
    masked_ids = token_ids.copy()
    labels = np.full_like(token_ids, -100)

    labels[mask_positions] = token_ids[mask_positions]

    condition_1 = mask_positions & (replace_probs < 0.8)
    condition_2 = mask_positions & (replace_probs >= 0.8) & (replace_probs < 0.9)
    condition_3 = mask_positions & (replace_probs >= 0.9)

    masked_ids[condition_1] = mask_token_id
    masked_ids[condition_2] = random_tokens[condition_2]
    masked_ids[condition_3] = token_ids[condition_3]

    return {"masked_ids": masked_ids, "labels": labels}
    