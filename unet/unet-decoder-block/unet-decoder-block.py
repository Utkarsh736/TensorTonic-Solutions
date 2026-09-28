import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray,
                       W_up: np.ndarray, b_up: np.ndarray,
                       kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 decoder features in NHWC layout.
    """
    def conv_relu(value, weights, bias):
        size = weights.shape[0]
        pad = size // 2
        padded = np.pad(value, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        output = np.empty((*value.shape[:3], weights.shape[3]), dtype=np.float64)
        for i in range(value.shape[1]):
            for j in range(value.shape[2]):
                patch = padded[:, i:i + size, j:j + size, :]
                output[:, i, j, :] = np.tensordot(patch, weights, axes=((1, 2, 3), (0, 1, 2))) + bias
        return np.maximum(output, 0.0)

    upsampled = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    projected = np.maximum(0.0, upsampled @ W_up + b_up)
    offset_h = (skip.shape[1] - projected.shape[1]) // 2
    offset_w = (skip.shape[2] - projected.shape[2]) // 2
    cropped = skip[:, offset_h:offset_h + projected.shape[1], offset_w:offset_w + projected.shape[2], :]
    merged = np.concatenate([cropped, projected], axis=-1)
    return conv_relu(conv_relu(merged, kernel1, bias1), kernel2, bias2)