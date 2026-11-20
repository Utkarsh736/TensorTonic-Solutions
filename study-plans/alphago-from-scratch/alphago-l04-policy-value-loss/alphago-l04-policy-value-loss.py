import torch
import torch.nn.functional as F

def alpha_go_policy_value_loss(
    policy_logits: torch.Tensor,
    predicted_values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    parameters: list,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns scalar float32 tensors: total, policy, value, and regularization loss.
    """
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

    value_loss = torch.mean((predicted_values - target_values) ** 2)

    if parameters:
        reg = sum(torch.sum(p * p) for p in parameters) * l2_coefficient
    else:
        reg = policy_logits.new_zeros(())

    total_loss = policy_loss + value_loss + reg

    return total_loss, policy_loss, value_loss, reg
