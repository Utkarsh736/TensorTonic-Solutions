#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<N){
        float val = input[idx];

        float exp_pos = __expf(val);
        float exp_neg = __expf(-val);
        
        output[idx] = (exp_pos-exp_neg)/(exp_pos+exp_neg);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}