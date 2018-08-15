#pragma once

namespace stk {
    class GpuVolume;
}

void find_min_max_1(stk::GpuVolume& vol, float& min, float& max);
void find_min_max_2(stk::GpuVolume& vol, float& min, float& max);
