import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    accumulator = 0.0
    for n in range(N):
        a_val = tl.load(a + row * stride_am + n * stride_an)
        b_val = tl.load(b + n * stride_bn + col * stride_bk)
        accumulator += a_val * b_val

    tl.store(c + row * stride_cm + col * stride_ck, accumulator)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (M, K)
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
    )
