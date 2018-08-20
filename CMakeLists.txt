cmake_minimum_required(VERSION 3.8)

project(stk-benchmark)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Do not build benchmark tests")
if(NOT WIN32)
    find_package(benchmark)
endif()
if(NOT benchmark_FOUND)
    add_subdirectory(benchmark)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(STK_USE_CUDA ON CACHE BOOL "" FORCE)
#set(STK_ENABLE_FAST_MATH OFF CACHE BOOL "")

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting default build type: 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (MSVC)
    add_definitions( -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_DEPRECATE )
    set(EXTRA_FLAGS "/FS")

    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    
    if (STK_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} /fp:fast")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(EXTRA_FLAGS "-Wall -fPIC")

    if (STK_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
elseif(CMAKE_CXX_COMPILER_ID matches "Clang")
    set(EXTRA_FLAGS "-Wall -fPIC")

    if (STK_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
endif()

if (STK_USE_CUDA)
    add_definitions(-DSTK_USE_CUDA)
    find_package(CUDA REQUIRED)
endif()

if(MSVC)
    # /wd4722 : destructor never returns, potential memory leak
    # /wd4244 : 'return': conversion from 'int64_t' to 'double',
    # /wd1000 : 'state': unreferenced formal parameter
    set(EXTRA_FLAGS "${EXTRA_FLAGS} /wd4722 /wd4244 /wd4100")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS} /wd4722")
    endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_LINK_FLAGS_RELEASE}")


set(STK_BUILD_EXAMPLES OFF CACHE BOOL "Do not build stk examples")

add_subdirectory(stk)
add_subdirectory(src)
