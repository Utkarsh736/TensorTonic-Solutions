#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sq[BLOCK_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* x_row = input + (size_t)row*N;
    float* y_row = output + (size_t)row * N;

    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (int j = tid; j<N; j += nthreads){
        float v = x_row[j];
        local_sum += v;
        local_sq += v*v;
    }

    s_sum[tid] = local_sum;
    s_sq[tid] = local_sq;
    __syncthreads();

    for (int stride = nthreads/2; stride>0; stride>>=1){
        if(tid<stride){
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }

        __syncthreads();
    }

    float mean = s_sum[0]/(float)N;
    float var = s_sq[0]/(float)N-mean*mean;
    float inv_std = rsqrtf(var + eps);

    __syncthreads();

    for(int j = tid; j<N; j+=nthreads){
        y_row[j] = (x_row[j]-mean)*inv_std*gamma[j] +  beta[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = BLOCK_SIZE;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
