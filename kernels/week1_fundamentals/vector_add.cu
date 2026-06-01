// vector_add.cu — Week 1, Day 1
// Element-wise add of two float arrays on the GPU: out[i] = a[i] + b[i].
//
// Build target: T4 (compute capability 7.5). On Kaggle P100 use sm_60.
// This file is consumed by load_inline (see test_vector_add.py), which wraps
// it with pybind11 and compiles it into a Python-importable .so via nvcc + ninja.

#include <torch/extension.h>

// ---- GPU code (runs on the device, one instance per thread) ----
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) {                                    // guard: extra threads do nothing
        out[i] = a[i] + b[i];                       // the actual work, one element
    }
}

// ---- CPU code (runs on the host, decides the launch and returns a tensor) ----
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(a.numel() == b.numel(), "inputs must have the same size");

    int n = a.numel();
    torch::Tensor out = torch::empty_like(a);       // fresh output (not in-place)

    int threads = 256;                              // threads per block
    int blocks  = (n + threads - 1) / threads;      // ceiling div -> cover all n

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n);

    return out;
}
