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
