import numpy as np

def crop_and_concat(encoder_features: np.ndarray,
                    decoder_features: np.ndarray) -> np.ndarray:
    """
    Returns the centered encoder crop concatenated with decoder features.
    """
    B, H_e, W_e, C_e = encoder_features.shape
    _, H_d, W_d, C_d = decoder_features.shape
    
    offset_h = (H_e - H_d) // 2
    offset_w = (W_e - W_d) // 2
    
    cropped = encoder_features[:, offset_h:offset_h + H_d, offset_w:offset_w + W_d, :]
    
    return np.concatenate([cropped, decoder_features], axis=-1)