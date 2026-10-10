import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int,
                            block_layers: list, compression: float) -> torch.Tensor:
    channels = stem_channels
    result = [channels]
    for index, layer_count in enumerate(block_layers):
        channels += layer_count * growth_rate
        result.append(channels)
        if index < len(block_layers) - 1:
            channels = math.floor(channels * compression)
            result.append(channels)
    return torch.tensor(result, dtype=torch.int64)