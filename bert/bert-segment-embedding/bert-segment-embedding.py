import numpy as np

def bert_embeddings(token_ids: np.ndarray, segment_ids: np.ndarray,
                    token_embeddings: np.ndarray, position_embeddings: np.ndarray,
                    segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Returns the float64 BERT input embeddings with shape (B, S, H).
    """
    B,S = token_ids.shape
    positions = np.arange(S)

    token_vector = token_embeddings[token_ids]
    position_vector = position_embeddings[positions]
    segment_vector = segment_embeddings[segment_ids]
    
    return token_vector+position_vector+segment_vector