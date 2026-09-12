import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols<n_cols

    x_row = x_ptr + row*x_row_stride
    out_row = out_ptr + row*out_row_stride

    x = tl.load(x_row+cols, mask=mask, other=-float("inf"))


    m = tl.max(x, axis=0)
    x = x-m
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    out = e/denom

    tl.store(out_row+cols, out, mask=mask)
    

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )