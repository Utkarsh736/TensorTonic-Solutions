import numpy as np

def conv2d_same(x, kernel, bias):

    B, H, W, C_in = x.shape
    K, _, _, C_out = kernel.shape

    pad_size = K // 2
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0))

    x_padded = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

    shape_out = (B, H, W, K, K, C_in)
    strides_out = (
        x_padded.strides[0],  # Batch stride
        x_padded.strides[1],  # Row stride
        x_padded.strides[2],  # Col stride
        x_padded.strides[1],  # Kernel row stride
        x_padded.strides[2],  # Kernel col stride
        x_padded.strides[3]   # Channel stride
    )

    windows = np.lib.stride_tricks.as_strided(x_padded, shape=shape_out, strides=strides_out)

    out = np.einsum('bhwklc,klco->bhwo', windows, kernel)

    out += bias  # shape: (B, H, W, C_out)

    return out

def relu(x):
  return np.maximum(x, 0)

def unet_bottleneck(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                    kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 bottleneck features in NHWC layout.
    """
    conv1 = conv2d_same(x, kernel1, bias1)
    act1 = relu(conv1)
    conv2 = conv2d_same(act1, kernel2, bias2)
    return relu(conv2)