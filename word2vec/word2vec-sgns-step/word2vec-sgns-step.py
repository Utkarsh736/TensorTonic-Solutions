import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    w_in = W_in.clone()
    w_out = W_out.clone()
    v_c = W_in[center_id].clone()
    op_snap = W_out.clone()

    pos_coeff = torch.sigmoid(torch.dot(v_c, op_snap[pos_id])) -1.0
    grad_c = pos_coeff * op_snap[pos_id]
    grad_op = {pos_id: pos_coeff * v_c}

    for neg_id in neg_ids.tolist():
        coeff = torch.sigmoid(torch.dot(v_c, op_snap[neg_id]))
        grad_c = grad_c + coeff * op_snap[neg_id]
        grad_op[neg_id] = grad_op.get(neg_id, torch.zeros_like(v_c)) + coeff*v_c

    w_in[center_id] = w_in[center_id] - lr * grad_c
    for word_id, grad in grad_op.items():
        w_out[word_id] = w_out[word_id] - lr * grad
        
    return {"W_in": w_in, "W_out": w_out}