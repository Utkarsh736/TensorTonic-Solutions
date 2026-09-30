import torch

def gradient_accumulation_step(
    param: torch.Tensor, microbatch_inputs: list[torch.Tensor],
    microbatch_targets: list[torch.Tensor], lr: float,
) -> dict:
    """
    Returns new_param and full_grad tensors in a dictionary.
    """
    N = sum(X.shape[0] for X in microbatch_inputs)
    w = param.detach().clone().requires_grad_(True)
    for m in range(len(microbatch_inputs)):
      inputs = microbatch_inputs[m]
      targets = microbatch_targets[m]

      predictions = inputs @ w
      loss = (predictions - targets).square().sum() / N
      loss.backward()

    new_param = param.detach() - lr * w.grad
    return {"new_param": new_param, "full_grad": w.grad}
