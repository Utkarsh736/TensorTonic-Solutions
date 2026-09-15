import torch

def noise_distribution(counts: torch.Tensor,
                       alpha: float = 0.75) -> torch.Tensor:
    """
    Returns the float64 negative-sampling distribution over the vocabulary.
    """
    counts = counts.to(torch.float64)

    pow_counts = torch.pow(counts, alpha)
    sum_counts = torch.sum(pow_counts)

    dist = pow_counts/sum_counts

    return dist