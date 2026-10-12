import torch
import torch.nn.functional as F

def bottleneck_layer(x: torch.Tensor, bn1_gamma: torch.Tensor, bn1_beta: torch.Tensor,
                     bn1_mean: torch.Tensor, bn1_var: torch.Tensor,
                     conv1_weight: torch.Tensor, bn2_gamma: torch.Tensor,
                     bn2_beta: torch.Tensor, bn2_mean: torch.Tensor,
                     bn2_var: torch.Tensor, conv2_weight: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet-B bottleneck transformation.
    """
    def batch_norm(values, gamma, beta, mean, var):
        gamma = gamma[None, :, None, None]
        mean  = mean[None, :, None, None]
        var = var[None, :, None, None]
        beta = beta[None, :, None, None]
        
        return gamma * (values - mean) / torch.sqrt(var + eps) + beta

    h = F.conv2d(F.relu(batch_norm(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var)), conv1_weight, bias=None)
    y = F.conv2d(F.relu(batch_norm(h, bn2_gamma, bn2_beta, bn2_mean, bn2_var)),
                        conv2_weight, bias=None, padding=1)

    return y