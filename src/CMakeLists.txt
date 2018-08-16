set(STK_BENCHMARK_SRCS 
    "bm_main.cpp"
    "bm_find_min_max.cpp"
    "bm_normalize.cpp"
    "test_data.cpp"
)


if (STK_USE_CUDA)
    set(STK_BENCHMARK_SRCS
        ${STK_BENCHMARK_SRCS}
        "gpu/find_min_max.cu"
        "gpu/find_min_max.h"
    )
endif ()

if (STK_USE_CUDA)
    cuda_add_executable(stk-bm ${STK_BENCHMARK_SRCS})
else ()
    add_executable(stk-bm ${STK_BENCHMARK_SRCS})
endif ()

target_link_libraries(stk-bm benchmark stk)

