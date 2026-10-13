import torch
import torch.nn.functional as F

def dense_block(x: torch.Tensor, layers: list, growth_rate: int,
                eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 concatenation of the block input and all layer outputs.
    """
    features = [x]

    for layer_dict in layers:
        current = torch.cat(features, dim=1)

        gamma = layer_dict["bn_gamma"][None, :, None, None]
        beta = layer_dict["bn_beta"][None, :, None, None]
        mean = layer_dict["bn_mean"][None, :, None, None]
        var = layer_dict["bn_var"][None, :, None, None]

        norm = gamma*(current-mean)/torch.sqrt(var+eps) + beta
        conv_weight = layer_dict["conv_weight"]
        h = F.conv2d(F.relu(norm), conv_weight, padding=1, bias=None)

        features.append(h)
        
    return torch.cat(features, dim=1)