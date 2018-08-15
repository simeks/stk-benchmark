#include <benchmark/benchmark.h>


static void BM_Test(benchmark::State& state)
{
    printf("asd");
}

BENCHMARK(BM_Test);