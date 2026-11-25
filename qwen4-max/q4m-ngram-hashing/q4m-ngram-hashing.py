import torch
import torch.nn.functional as F

def ngram_hash_ids(token_ids: torch.Tensor, pad_id: int, multipliers: torch.Tensor, prime_sizes: list[list[int]]) -> torch.Tensor:
    """
    Returns the int64 hash ids of every suffix n-gram head at every position.
    """
    N = multipliers.shape[0]                     
    B, T = token_ids.shape

    padded = F.pad(token_ids, (N-1, 0), value=pad_id)

    mix = (padded[:, N-1:] * multipliers[0]) ^ (padded[:, N-2:-1] * multipliers[1])
    mixes = [mix]                                
    for n in range(3, N+1):
        mix = mix ^ (padded[:, N-n:-n+1] * multipliers[n-1])
        mixes.append(mix)

    head_results = []
    for order_idx, mix in enumerate(mixes):
        for prime in prime_sizes[order_idx]:
            head_results.append(torch.remainder(mix, prime))

    return torch.stack(head_results, dim=-1)
