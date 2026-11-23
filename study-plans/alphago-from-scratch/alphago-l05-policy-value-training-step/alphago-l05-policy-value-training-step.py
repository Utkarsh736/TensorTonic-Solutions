import torch
import torch.nn.functional as F

def policy_value_training_step(
    features: torch.Tensor,
    policy_weight: torch.Tensor,
    policy_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns detached float32 tensors: loss, policy weight/bias, value weight/bias.
    """
    X = features.clone().detach().to(torch.float32)
    Wp = policy_weight.clone().detach().to(torch.float32).requires_grad_(True)
    bp = policy_bias.clone().detach().to(torch.float32).requires_grad_(True)
    wv = value_weight.clone().detach().to(torch.float32).requires_grad_(True)
    bv = value_bias.clone().detach().to(torch.float32).requires_grad_(True)

    target_p = target_policies.clone().detach().to(torch.float32)
    target_v = target_values.clone().detach().to(torch.float32)

    logits = X @ Wp.T + bp                         # (B, A)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_p * log_probs).sum(dim=1).mean()

    v = torch.tanh(X @ wv + bv)                    # (B,)
    value_loss = ((v - target_v) ** 2).mean()

    reg = l2_coefficient * (
        Wp.pow(2).sum()
        + bp.pow(2).sum()
        + wv.pow(2).sum()
        + bv.pow(2).sum()
    )

    total_loss = policy_loss + value_loss + reg

    grads = torch.autograd.grad(total_loss, [Wp, bp, wv, bv])

    with torch.no_grad():
        Wp_new = Wp - learning_rate * grads[0]
        bp_new = bp - learning_rate * grads[1]
        wv_new = wv - learning_rate * grads[2]
        bv_new = bv - learning_rate * grads[3]

    return (
        total_loss.detach(),
        Wp_new.detach(),
        bp_new.detach(),
        wv_new.detach(),
        bv_new.detach(),
    )
