#include "find_min_max.h"

#include <stk/cuda/cuda.h>
#include <stk/cuda/ptr.h>

#include <stk/image/gpu_volume.h>

#include <algorithm>

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


// Implementation 2
// Unrolled loop with sequential addressing

__global__ void algo2_reduce_volume_min_max(
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

    shared[tid].x = FLT_MAX;
    shared[tid].y = -FLT_MAX;

    if (x < dims.x &&
        y < dims.y &&
        z < dims.z) {
        shared[tid].x = in(x,y,z);
        shared[tid].y = in(x,y,z);
    }
    __syncthreads();

    // Block size is always 512 for first pass of the reduction 
    //  (with volume as input)

    if (tid < 256) { 
        shared[tid].x = min(shared[tid].x, shared[tid+256].x);
        shared[tid].y = max(shared[tid].y, shared[tid+256].y);
    }
    __syncthreads();

    if (tid < 128) {
        shared[tid].x = min(shared[tid].x, shared[tid+128].x);
        shared[tid].y = max(shared[tid].y, shared[tid+128].y);
    }
    __syncthreads();

    if (tid < 64) {
        shared[tid].x = min(shared[tid].x, shared[tid+64].x);
        shared[tid].y = max(shared[tid].y, shared[tid+64].y);
    }
    __syncthreads();

    if (tid < 32) {
        shared[tid].x = min(shared[tid].x, shared[tid+32].x);
        shared[tid].y = max(shared[tid].y, shared[tid+32].y);
    }
    __syncthreads();

    if (tid < 16) {
        shared[tid].x = min(shared[tid].x, shared[tid+16].x);
        shared[tid].y = max(shared[tid].y, shared[tid+16].y);
    }
    __syncthreads();
    
    if (tid < 8) {
        shared[tid].x = min(shared[tid].x, shared[tid+8].x);
        shared[tid].y = max(shared[tid].y, shared[tid+8].y);
    }
    __syncthreads();
    
    if (tid < 4) {
        shared[tid].x = min(shared[tid].x, shared[tid+4].x);
        shared[tid].y = max(shared[tid].y, shared[tid+4].y);
    }
    __syncthreads();

    if (tid < 2) {
        shared[tid].x = min(shared[tid].x, shared[tid+2].x);
        shared[tid].y = max(shared[tid].y, shared[tid+2].y);
    }
    __syncthreads();
    
    if (tid == 0) {
        // Write min/max for block to output
        out[bid].x = min(shared[tid].x, shared[tid+1].x);
        out[bid].y = max(shared[tid].y, shared[tid+1].y);
    }
}

template<int BLOCK_SIZE>
__global__ void algo2_reduce_min_max(
    unsigned int n,
    float2* in,
    float2* out)
{
    extern __shared__ float2 shared[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + threadIdx.x; // global

    if (i < n) shared[tid] = in[i];
    else shared[tid] = {FLT_MAX, -FLT_MAX};

    if (i + BLOCK_SIZE < n) {
        shared[tid].x = min(shared[tid].x, in[i+BLOCK_SIZE].x);
        shared[tid].y = max(shared[tid].y, in[i+BLOCK_SIZE].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >= 512 && tid < 256) {
        shared[tid].x = min(shared[tid].x, shared[tid+256].x);
        shared[tid].y = max(shared[tid].y, shared[tid+256].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >= 256 && tid < 128) {
        shared[tid].x = min(shared[tid].x, shared[tid+128].x);
        shared[tid].y = max(shared[tid].y, shared[tid+128].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >= 128 && tid < 64) {
        shared[tid].x = min(shared[tid].x, shared[tid+64].x);
        shared[tid].y = max(shared[tid].y, shared[tid+64].y);
    }
    __syncthreads();

    // fully unroll reduction within a single warp
    if (BLOCK_SIZE >=  64 && tid < 32) {
        shared[tid].x = min(shared[tid].x, shared[tid+32].x);
        shared[tid].y = max(shared[tid].y, shared[tid+32].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >=  32 && tid < 16) {
        shared[tid].x = min(shared[tid].x, shared[tid+16].x);
        shared[tid].y = max(shared[tid].y, shared[tid+16].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >=  16 && tid < 8) {
        shared[tid].x = min(shared[tid].x, shared[tid+8].x);
        shared[tid].y = max(shared[tid].y, shared[tid+8].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >= 8 && tid < 4) {
        shared[tid].x = min(shared[tid].x, shared[tid+4].x);
        shared[tid].y = max(shared[tid].y, shared[tid+4].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >= 4 && tid < 2) {
        shared[tid].x = min(shared[tid].x, shared[tid+2].x);
        shared[tid].y = max(shared[tid].y, shared[tid+2].y);
    }
    __syncthreads();

    if (BLOCK_SIZE >= 2 && tid < 1) {
        shared[tid].x = min(shared[tid].x, shared[tid+1].x);
        shared[tid].y = max(shared[tid].y, shared[tid+1].y);
    }
    __syncthreads();

    if (tid == 0) out[blockIdx.x] = shared[tid];
}
void algo2_run_kernel(uint32_t n_blocks, uint32_t n_threads, uint32_t n, float2* d_in, float2* d_out)
{
    dim3 grid_size {n_blocks, 1, 1};
    dim3 block_size {n_threads, 1, 1};

    uint32_t shared_size = 2*sizeof(float)*n_threads;

    switch (n_threads) {
        case 512:
            algo2_reduce_min_max<512><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case 256:
            algo2_reduce_min_max<256><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case 128:
            algo2_reduce_min_max<128><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case 64:
            algo2_reduce_min_max<64><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case 32:
            algo2_reduce_min_max<32><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case 16:
            algo2_reduce_min_max<16><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case  8:
            algo2_reduce_min_max<8><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case  4:
            algo2_reduce_min_max<4><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
        case  2:
            algo2_reduce_min_max<2><<<grid_size, block_size, shared_size>>>(n, d_in, d_out);
            break;
    };
}

void find_min_max_2(stk::GpuVolume& vol, float& min, float& max)
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

    algo2_reduce_volume_min_max<<<grid_size, block_size, 
        uint32_t(2*sizeof(float)*512)>>>(
        vol, vol.size(), d_out
    );

    CUDA_CHECK_ERRORS(cudaPeekAtLastError());
    CUDA_CHECK_ERRORS(cudaDeviceSynchronize());

    while (n > 1) {
        // block_count should always be pow2 as it follows the gridsize from 
        //  previous step
        uint32_t n_threads = std::min<uint32_t>(n, 512);
        uint32_t n_blocks = (n + n_threads - 1) / n_threads;

        CUDA_CHECK_ERRORS(cudaMemcpy(d_in, d_out, 2*n*sizeof(float), 
            cudaMemcpyDeviceToDevice));

        algo2_run_kernel(n_blocks, n_threads, n, d_in, d_out);

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