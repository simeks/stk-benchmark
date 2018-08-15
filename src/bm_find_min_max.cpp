#include <benchmark/benchmark.h>

#include <stk/image/gpu_volume.h>
#include <stk/image/volume.h>

#include "gpu/find_min_max.h"

#include <random>

namespace {
    int _data_seed = 4321;
}

void fill_data(float* d, size_t len) {
    std::mt19937 gen(_data_seed);
    std::uniform_int_distribution<> dis(0, 10000000);

    for (size_t i = 0; i < len; ++i) {
        d[i] = (float)dis(gen);
    }
}


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

// Algo 1
static void BM_gpu_volume_find_min_max_1(benchmark::State& state)
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

    for (auto _ : state) {
        
        float min, max;
        find_min_max_1(gpu_vol, min, max);
    }
}

// Algo 2
static void BM_gpu_volume_find_min_max_2(benchmark::State& state)
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

    for (auto _ : state) {
        
        float min, max;
        find_min_max_2(gpu_vol, min, max);
    }
}


BENCHMARK(BM_volume_find_min_max)->Range(8, 512);
BENCHMARK(BM_gpu_volume_find_min_max_1)->Range(8, 512);
BENCHMARK(BM_gpu_volume_find_min_max_2)->Range(8, 512);
