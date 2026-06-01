"""
test_vector_add.py — Week 1, Day 1

Compiles vector_add.cu on the fly, checks correctness against torch.add,
and benchmarks memory bandwidth (GB/s) with CUDA events.

Run:  python test_vector_add.py
T4 theoretical peak bandwidth ~= 320 GB/s. A clean vector_add lands ~70-90% of that.
"""
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

# T4 = compute capability 7.5. Kaggle P100 = 6.0 -> change sm_75 below to sm_60.
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

# Read the kernel source from the .cu file (keeps CUDA in one place).
CUDA_SRC = Path(__file__).with_name("vector_add.cu").read_text()
CPP_SRC = "torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);"

mod = load_inline(
    name="vector_add_mod",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["vector_add"],
    verbose=True,
    with_cuda=True,
    # Explicit arch: avoids the CUDA 12.8 auto-detect ImportError on Colab.
    extra_cuda_cflags=["-gencode", "arch=compute_75,code=sm_75"],
)


def test_correctness():
    print("== correctness ==")
    all_ok = True
    for n in [1024, 1_000_000, 10_000_000]:
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        mine = mod.vector_add(a, b)
        ref = torch.add(a, b)
        ok = torch.allclose(mine, ref)
        all_ok &= ok
        print(f"  n={n:>10}  match={ok}")
    assert all_ok, "correctness FAILED"
    print("  all correct\n")


def benchmark(n, iters=100):
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")

    for _ in range(10):              # warmup (first call pays one-time cost)
        mod.vector_add(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        mod.vector_add(a, b)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    bytes_moved = 3 * n * 4          # read a, read b, write out; 4 bytes/float32
    gbps = bytes_moved / (ms / 1000) / 1e9
    pct = gbps / 320 * 100
    print(f"  n={n:>10}  {ms:.4f} ms/call   {gbps:6.1f} GB/s   ({pct:.0f}% of T4 peak)")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "\n")
    test_correctness()
    print("== bandwidth ==")
    for n in [1_000_000, 10_000_000, 50_000_000]:
        benchmark(n)
