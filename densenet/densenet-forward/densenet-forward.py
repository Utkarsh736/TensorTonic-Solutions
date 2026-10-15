import torch
import torch.nn.functional as F

def densenet_forward(x: torch.Tensor, weights: dict, growth_rate: int,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 class logits from the complete DenseNet forward pass.
    """
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

    stem_w = weights["stem_conv"]
    z = F.conv2d(x, stem_w, bias=None, stride=1, padding=1)

    blocks = weights["blocks"]
    transitions = weights["transitions"]
    for i, block_layer in enumerate(blocks):
        z = dense_block(z, block_layer, growth_rate, eps)
        if i < len(transitions):
            t = transitions[i]
            z = transition_layer(z, t["bn_gamma"],
                                 t["bn_beta"],
                                 t["bn_mean"],
                                 t["bn_var"],
                                 t["conv_weight"],
                                eps
                                )

    gamma = weights["final_bn_gamma"][None, :, None, None]
    beta = weights["final_bn_beta"][None, :, None, None]
    mean = weights["final_bn_mean"][None, :, None, None]
    var = weights["final_bn_var"][None, :, None, None]
    z = F.relu(gamma * (z - mean) / torch.sqrt(var + eps) + beta)
    
    p = z.mean(dim=(2,3))

    logits = p@weights["fc_weight"].T + weights["fc_bias"]

    return logits