import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_mask = offs_m < M

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < N

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + n_offs[None, :] * stride_an
        a_tile = tl.load(a_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

        x_chunk = tl.load(x_ptr + n_offs, mask=n_mask, other=0.0)

        acc += tl.sum(a_tile * x_chunk[None, :], axis=1)

    tl.store(out_ptr + offs_m, acc, mask=m_mask)


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )