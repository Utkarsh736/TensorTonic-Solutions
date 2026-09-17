import numpy as np

def vgg_forward(x: np.ndarray, config: list, kernels: list,
                biases: list, classifier: dict) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    out = x
    k_idx = 0

    for entry in config:
        if entry == 'M':
            N, H, W, C = out.shape
            out = out.reshape(N, H//2, 2, W//2, 2, C)
            out = out.max(axis=(2, 4))

        else:
            kernel = kernels[k_idx]
            bias = biases[k_idx]
            k_idx += 1

            N, H, W, C_in = out.shape
            kH, kW, _, C_out = kernel.shape

            pad_h = (kH - 1)//2
            pad_w = (kW - 1)//2

            out_padded = np.pad(
                out,
                ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode='constant',
            )

            conv_out = np.zeros((N, H, W, C_out))
            for i in range(H):
                for j in range(W):
                    x_slice = out_padded[:, i:i+kH, j:j+kW, :]
                    conv_out[:, i, j, :] = np.tensordot(x_slice, kernel, axes=((1,2,3), (0,1,2))) + bias

            out = np.maximum(0.0, conv_out)


    B = out.shape[0]
    flat_out = out.reshape(B, -1)

    fc1 = np.dot(flat_out, np.asarray(classifier['W1'])) + np.asarray(classifier['b1'])
    fc1 = np.maximum(0.0, fc1)

    fc2 = np.dot(fc1, np.asarray(classifier['W2'])) + np.asarray(classifier['b2'])
    fc2 = np.maximum(0.0, fc2)

    logits = np.dot(fc2, np.asarray(classifier['W3'])) + np.asarray(classifier['b3'])

    return logits