import torch
import torch.nn.functional as F

def transition_layer(x: torch.Tensor, bn_gamma: torch.Tensor, bn_beta: torch.Tensor,
                     bn_mean: torch.Tensor, bn_var: torch.Tensor,
                     conv_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet transition layer.
    """
    gamma = bn_gamma[None, :, None, None]
    beta = bn_beta[None, :, None, None]
    mean = bn_mean[None, :, None, None]
    var = bn_var[None, :, None, None]

    # Norm
    norm = gamma*(x-mean)/torch.sqrt(var+eps) + beta
    y = F.conv2d(F.relu(norm), conv_weight, padding=0, bias=None)

    return F.avg_pool2d(y, kernel_size=2, stride=2)