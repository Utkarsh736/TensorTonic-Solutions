#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<N){
        float sqr_rt = sqrtf(2.0);
        float val = erff(input[idx]/sqr_rt);

        output[idx] = 0.5*input[idx]*(1 + val);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
