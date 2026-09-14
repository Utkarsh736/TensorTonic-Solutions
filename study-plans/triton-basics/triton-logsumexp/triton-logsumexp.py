import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    r_id = tl.program_id(axis=0)
    r_ptr = x_ptr + r_id*x_row_stride

    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs<n_cols

    x = tl.load(r_ptr+col_offs, mask=mask, other=-float("inf"))

    r_max = tl.max(x, axis=0)

    safe_x = x-r_max
    exp_x = tl.exp(safe_x)

    exp_x = tl.where(mask, exp_x, 0.0)
    r_sum = tl.sum(exp_x, axis=0)

    lse = tl.math.log(r_sum) + r_max

    out_row_ptr = out_ptr + r_id
    tl.store(out_row_ptr, lse)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )