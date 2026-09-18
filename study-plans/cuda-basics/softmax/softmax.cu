#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // max
    float local_max = -INFINITY;
    for (int i=tid; i<N; i+=nthreads){
        local_max = fmaxf(local_max, input[i]);
    }
    shared[tid] = local_max;
    __syncthreads();


    for (int stride = nthreads/2; stride>0; stride>>=1){
        if(tid<stride){
            shared[tid] = fmaxf(shared[tid], shared[tid+stride]);
        }

        __syncthreads();
    }

    float m = shared[0];
    __syncthreads();

    // exp(x-m) and sum
    float local_sum = 0.0f;
    for (int i = tid; i<N; i+=nthreads){
        float e = expf(input[i]-m);
        output[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for(int stride = nthreads/2; stride>0; stride >>= 1){
        if(tid<stride){
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float S = shared[0];

    for(int i = tid; i<N; i += nthreads){
        output[i]/=S;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    // int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<1, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}