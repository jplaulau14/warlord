import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        rn = n + tl.arange(0, BLOCK_N)

        a_ptrs = a + rm[:, None] * stride_am + rn[None, :] * stride_an
        b_ptrs = b + rn[:, None] * stride_bn + rk[None, :] * stride_bk

        a_tile = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
        b_tile = tl.load(b_ptrs, mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)

        acc += tl.dot(a_tile, b_tile)

    c_ptrs = c + rm[:, None] * stride_cm + rk[None, :] * stride_ck
    mask = (rm[:, None] < M) & (rk[None, :] < K)
    tl.store(c_ptrs, acc, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 32, 64
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
