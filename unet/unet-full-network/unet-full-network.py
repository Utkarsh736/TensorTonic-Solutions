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

def maxpool2x2(x):
    B, H, W, C = x.shape
    
    # Ensure H and W are even as requested
    assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even numbers."
    
    reshaped_x = x.reshape(B, H // 2, 2, W // 2, 2, C)
    out = np.max(reshaped_x, axis=(2, 4))
    
    return out

def unet_encoder_block(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> dict:
    """
    Returns pooled and skip as float64 arrays in a dictionary.
    """
    conv1 = conv2d_same(x, kernel1, bias1)
    act1 = relu(conv1)
    
    conv2 = conv2d_same(act1, kernel2, bias2)
    skip = relu(conv2)  
    
    pooled = maxpool2x2(skip)
    
    return {
        "skip": skip,       
        "pooled": pooled    
    }

def unet_decoder_block(x, skip, W_up, b_up, kernel1, bias1, kernel2, bias2):
    upsampled = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
    projected = np.maximum(0.0, upsampled @ W_up + b_up)

    offset_h = (skip.shape[1] - projected.shape[1]) // 2
    offset_w = (skip.shape[2] - projected.shape[2]) // 2
    cropped = skip[:, offset_h:offset_h + projected.shape[1],
                    offset_w:offset_w + projected.shape[2], :]

    merged = np.concatenate([cropped, projected], axis=-1)
    return relu(conv2d_same(relu(conv2d_same(merged, kernel1, bias1)), kernel2, bias2))

def unet_bottleneck(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                    kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 bottleneck features in NHWC layout.
    """
    conv1 = conv2d_same(x, kernel1, bias1)
    act1 = relu(conv1)
    conv2 = conv2d_same(act1, kernel2, bias2)
    return relu(conv2)

def unet_output_layer(features: np.ndarray, W_out: np.ndarray,
                b_out: np.ndarray) -> np.ndarray:
    """
    Returns float64 per-pixel class logits in NHWC layout.
    """
    return features@W_out + b_out

def unet_forward(x: np.ndarray, weights: dict) -> np.ndarray:
    """
    Returns float64 per-pixel segmentation logits in NHWC layout.
    """
    encoder = unet_encoder_block(x, weights["enc_kernel1"],
                                 weights["enc_bias1"],
                                 weights["enc_kernel2"],
                                 weights["enc_bias2"])
    
    bridge = unet_bottleneck(encoder["pooled"],
                             weights["bridge_kernel1"],
                             weights["bridge_bias1"],
                             weights["bridge_kernel2"],
                             weights["bridge_bias2"])
    
    decoder = unet_decoder_block(bridge,
                                 encoder["skip"],
                                 weights["W_up"],
                                 weights["b_up"],
                                 weights["dec_kernel1"],
                                 weights["dec_bias1"],
                                 weights["dec_kernel2"],
                                 weights["dec_bias2"])

    
    return unet_output_layer(decoder, weights["W_out"], weights["b_out"])