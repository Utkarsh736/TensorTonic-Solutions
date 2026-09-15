import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    pairs = []
    seq_len = token_ids.size(0)

    for i in range(seq_len):
        start = max(0, i-window)
        end = min(seq_len, i+window+1)

        for j in range(start, end):
            if i!=j:
                pairs.append([token_ids[i].item(), token_ids[j].item()])


    if not pairs:
        return torch.empty((0,2), dtype=torch.int64)

    return torch.tensor(pairs, dtype=torch.int64)