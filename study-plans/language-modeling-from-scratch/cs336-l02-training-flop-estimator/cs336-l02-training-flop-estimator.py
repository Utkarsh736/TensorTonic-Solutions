def flop_estimator(matmuls: list[list[int]], attention_flops: int = 0) -> dict:
    """
    Returns integer forward_flops, backward_flops, and total_flops in a dictionary.
    """
    
    matmul_sum = 0
    for b,d,k in matmuls:
      matmul_sum += 2*b*d*k

    forward = matmul_sum + attention_flops
    backward = 2*forward
    total = forward + backward

    return {"forward_flops": forward, "backward_flops": backward, "total_flops": total}
