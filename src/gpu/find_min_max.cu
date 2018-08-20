#include "find_min_max.h"

#include <stk/cuda/cuda.h>
#include <stk/cuda/ptr.h>

#include <stk/image/gpu_volume.h>

#include <algorithm>
#include <cfloat>

namespace cuda = stk::cuda;


// Implementation 1
// Interleaved addressing w/o any optimization

__global__ void algo1_reduce_volume_min_max(
    const cuda::VolumePtr<float> in,
    dim3 dims,
    float2* out)
{
    extern __shared__ float2 shared[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    shared[tid] = {FLT_MAX, -FLT_MAX};

    if (x < dims.x &&
        y < dims.y &&
        z < dims.z) {
        shared[tid].x = in(x,y,z);
        shared[tid].y = in(x,y,z);
    }
    __syncthreads();

    for (unsigned int s=1; s < blockDim.x*blockDim.y*blockDim.z; s *= 2)
    {
        if ((tid % (2*s)) == 0)
        {
            shared[tid].x = min(shared[tid].x, shared[tid + s].x);
            shared[tid].y = max(shared[tid].y, shared[tid + s].y);
        }

        __syncthreads();
    }

    if (tid == 0) out[bid] = shared[0];
}

__global__ void algo1_reduce_min_max(
    unsigned int n,
    float2* in,
    float2* out)
{
    extern __shared__ float2 shared[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;

    if (i < n) shared[tid] = in[i];
    else shared[tid] = {FLT_MAX, -FLT_MAX};

    __syncthreads();

    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        if ((tid % (2*s)) == 0)
        {
            shared[tid].x = min(shared[tid].x, shared[tid + s].x);
            shared[tid].y = max(shared[tid].y, shared[tid + s].y);
        }

        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = shared[0];
}


void find_min_max_1(stk::GpuVolume& vol, float& min, float& max)
{
    dim3 block_size{8,8,8};

    dim3 grid_size {
        (vol.size().x + block_size.x - 1) / block_size.x,
        (vol.size().y + block_size.y - 1) / block_size.y,
        (vol.size().z + block_size.z - 1) / block_size.z
    };

    // Number of blocks (or values in the active buffer)
    uint32_t n = grid_size.x * grid_size.y * grid_size.z;

    // Allocate our global buffers
    float2* d_out;
    CUDA_CHECK_ERRORS(cudaMalloc(&d_out, 2*n*sizeof(float)));

    float2* d_in;
    CUDA_CHECK_ERRORS(cudaMalloc(&d_in, 2*n*sizeof(float)));

    algo1_reduce_volume_min_max<<<grid_size, block_size, 
        uint32_t(2*sizeof(float)*512)>>>(
        vol, vol.size(), d_out
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

    while (n > 1) {
        // block_count should always be pow2 as it follows the gridsize from 
        //  previous step
        uint32_t n_threads = std::min<uint32_t>(n, 1024);
        uint32_t n_blocks = (n + n_threads - 1) / n_threads;

        CUDA_CHECK_ERRORS(cudaMemcpy(d_in, d_out, 2*n*sizeof(float), 
            cudaMemcpyDeviceToDevice));

        algo1_reduce_min_max<<<{n_blocks,1,1}, {n_threads,1,1}, 
                                uint32_t(2*sizeof(float)*n_threads)>>>(
            n, d_in, d_out);

        CUDA_CHECK_ERRORS(cudaPeekAtLastError());
        CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

        n = n_blocks;
    }

    float2 min_max;
    CUDA_CHECK_ERRORS(cudaMemcpy(&min_max, d_out, 2*sizeof(float), cudaMemcpyDeviceToHost));

    min = min_max.x;
    max = min_max.y;

    CUDA_CHECK_ERRORS(cudaFree(d_in));
    CUDA_CHECK_ERRORS(cudaFree(d_out));
}

