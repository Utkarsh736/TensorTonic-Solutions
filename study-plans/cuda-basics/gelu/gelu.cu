#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<N){
        float x = input[idx];

        float cube = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * cube);

        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
