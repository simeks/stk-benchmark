#include <benchmark/benchmark.h>

#include <stk/filters/normalize.h>
#include <stk/filters/gpu/normalize.h>
#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

void fill_data(float* d, size_t len);

static void BM_volume_normalize(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    stk::VolumeFloat out;
    for (auto _ : state) {
        out = stk::normalize(vol, 0.0f, 1.0f);
    }
}

static void BM_volume_normalize_in_place(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    for (auto _ : state) {
        stk::normalize(vol, 0.0f, 1.0f, &vol);
    }
}


static void BM_gpu_volume_normalize_pitched(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    stk::GpuVolume gpu_vol(vol, stk::gpu::Usage_PitchedPointer);

    stk::GpuVolume out;
    for (auto _ : state) {
        out = stk::gpu::normalize(gpu_vol, 0.0f, 1.0f);
    }
}

static void BM_gpu_volume_normalize_in_place_pitched(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    stk::GpuVolume gpu_vol(vol, stk::gpu::Usage_PitchedPointer);

    for (auto _ : state) {
        stk::gpu::normalize(gpu_vol, 0.0f, 1.0f, &gpu_vol);
    }
}

static void BM_gpu_volume_normalize_texture(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    stk::GpuVolume gpu_vol(vol, stk::gpu::Usage_Texture);

    stk::GpuVolume out;
    for (auto _ : state) {
        out = stk::gpu::normalize(gpu_vol, 0.0f, 1.0f);
    }
}

static void BM_gpu_volume_normalize_in_place_texture(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    stk::GpuVolume gpu_vol(vol, stk::gpu::Usage_Texture);

    for (auto _ : state) {
        stk::gpu::normalize(gpu_vol, 0.0f, 1.0f, &gpu_vol);
    }
}

static void BM_gpu_volume_normalize_blocksize(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    dim3 block_size {
        (uint32_t)state.range(1),
        (uint32_t)state.range(2),
        1
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    stk::GpuVolume gpu_vol(vol, stk::gpu::Usage_PitchedPointer);

    stk::GpuVolume out;
    for (auto _ : state) {
        out = stk::gpu::normalize(gpu_vol, 0.0f, 1.0f, nullptr, block_size);
    }
}


BENCHMARK(BM_volume_normalize)->Range(8, 512);
BENCHMARK(BM_volume_normalize_in_place)->Range(8, 512);

BENCHMARK(BM_gpu_volume_normalize_pitched)->Range(8, 512);
BENCHMARK(BM_gpu_volume_normalize_in_place_pitched)->Range(8, 512);
BENCHMARK(BM_gpu_volume_normalize_texture)->Range(8, 512);
BENCHMARK(BM_gpu_volume_normalize_in_place_texture)->Range(8, 512);

// BENCHMARK(BM_gpu_volume_normalize_blocksize)->
//     Ranges({{8, 512}, {1, 32}, {1, 32}});
