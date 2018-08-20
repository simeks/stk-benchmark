#include <benchmark/benchmark.h>

#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include "gpu/find_min_max.h"

void fill_data(float* d, size_t len);

static void BM_volume_find_min_max(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);

    for (auto _ : state) {
        float min, max;
        stk::find_min_max(vol, min, max);
    }
}

// Non-optimized
static void BM_gpu_volume_find_min_max_slow(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);
    float gt_min, gt_max;
    stk::find_min_max(vol, gt_min, gt_max);

    stk::GpuVolume gpu_vol(vol);

    float min, max;
    for (auto _ : state) {
        find_min_max_1(gpu_vol, min, max);
    }
}

// STK version
static void BM_gpu_volume_find_min_max(benchmark::State& state)
{
    dim3 dims {
        (uint32_t)state.range(0),
        (uint32_t)state.range(0),
        (uint32_t)state.range(0)
    };

    stk::VolumeFloat vol(dims);
    fill_data((float*)vol.ptr(), dims.x*dims.y*dims.z);
    float gt_min, gt_max;
    stk::find_min_max(vol, gt_min, gt_max);

    stk::GpuVolume gpu_vol(vol);

    float min, max;
    for (auto _ : state) {
        stk::find_min_max(gpu_vol, min, max);
    }
}

BENCHMARK(BM_volume_find_min_max)
    ->RangeMultiplier(2)
    ->Range(8, 512);
BENCHMARK(BM_gpu_volume_find_min_max_slow)
    ->RangeMultiplier(2)
    ->Range(8, 512);
BENCHMARK(BM_gpu_volume_find_min_max)
    ->RangeMultiplier(2)
    ->Range(8, 512);
