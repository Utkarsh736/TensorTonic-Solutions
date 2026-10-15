import torch
import torch.nn.functional as F


def _bn_relu(x, gamma, beta, mean, var, eps):
    """Evaluation-mode BatchNorm + ReLU. Shape-preserving."""
    g = gamma[None, :, None, None]
    b = beta[None, :, None, None]
    m = mean[None, :, None, None]
    v = var[None, :, None, None]
    return F.relu(g * (x - m) / torch.sqrt(v + eps) + b)


def _dense_block(x, layers, eps):
    """Concatenate all previous outputs, transform, append. Returns the full stack."""
    features = [x]
    for layer in layers:
        h = _bn_relu(torch.cat(features, dim=1),
                     layer["bn_gamma"], layer["bn_beta"],
                     layer["bn_mean"], layer["bn_var"], eps)
        features.append(F.conv2d(h, layer["conv_weight"], bias=None, padding=1))
    return torch.cat(features, dim=1)


def _transition(x, t, eps):
    """Compress channels with a 1x1 conv, then halve H, W with 2x2 avg pool."""
    h = _bn_relu(x, t["bn_gamma"], t["bn_beta"],
                 t["bn_mean"], t["bn_var"], eps)
    z = F.conv2d(h, t["conv_weight"], bias=None, padding=0)
    return F.avg_pool2d(z, kernel_size=2, stride=2)


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    z = F.conv2d(x, weights["stem_conv"], bias=None, stride=1, padding=1)

    transitions = weights["transitions"]
    for i, block in enumerate(weights["blocks"]):
        z = _dense_block(z, block, eps)
        if i < len(transitions):
            z = _transition(z, transitions[i], eps)

    z = _bn_relu(z,
                 weights["final_bn_gamma"], weights["final_bn_beta"],
                 weights["final_bn_mean"], weights["final_bn_var"], eps)

    p = z.mean(dim=(2, 3))
    return p @ weights["fc_weight"].T + weights["fc_bias"]