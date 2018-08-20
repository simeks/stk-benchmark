#pragma once

namespace stk {
    class GpuVolume;
}

// Non-optimized naive version of find_min_max
void find_min_max_1(stk::GpuVolume& vol, float& min, float& max);
