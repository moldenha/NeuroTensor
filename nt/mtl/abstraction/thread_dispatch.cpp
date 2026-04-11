
#include "mtl_macros.h"
#include "thread_dispatch.h"
#include "mtl_context.h"
#include <algorithm>

namespace nt::mtl::abs{

ThreadDispatchConfig computeThreadDispatchConfig(int64_t N) {
    // Get max total threads allowed in a single threadgroup
    auto& mtl = MetalContext::instance();
    MTL::Size maxSize = mtl.device()->maxThreadsPerThreadgroup();
    uint64_t maxThreadsPerGroup =
    uint64_t(maxSize.width) * uint64_t(maxSize.height) * uint64_t(maxSize.depth);
    uint32_t max1DThreads = maxSize.width;


    if(N <= uint64_t(max1DThreads)) {
        // Pick a reasonable power-of-two size <= max
        uint32_t groupSize = 1;
        while (groupSize * 2 <= max1DThreads && groupSize * 2 <= N)
            groupSize *= 2;

        uint32_t numGroups = (N + groupSize - 1) / groupSize;

        return ThreadDispatchConfig {
            .gridSize = MTL::Size::Make(N, 1, 1),
            .threadgroupSize = MTL::Size::Make(groupSize, 1, 1)
        };
    }else {
        // split across 3D grid
        ThreadDispatchConfig config;
        uint64_t threadsRemaining = N;

        uint32_t gridX = std::min<uint64_t>(threadsRemaining, UINT32_MAX);
        threadsRemaining = (threadsRemaining + gridX - 1) / gridX;

        uint32_t gridY = std::min<uint64_t>(threadsRemaining, UINT32_MAX);
        threadsRemaining = (threadsRemaining + gridY - 1) / gridY;

        uint32_t gridZ = std::min<uint64_t>(threadsRemaining, UINT32_MAX);

        config.gridSize = MTL::Size::Make(gridX, gridY, gridZ);

        // Pick a threadgroup size in 3D (tgX * tgY * tgZ <= maxThreadsPerGroup)
        uint32_t tgX = 1, tgY = 1, tgZ = 1;

        // maximize tgX
        while (tgX * 2 <= gridX && tgX * 2 <= maxTG.width) tgX *= 2;
        // maximize tgY
        while (tgX * tgY * 2 <= maxThreadsPerGroup && tgY * 2 <= gridY && tgY * 2 <= maxTG.height) tgY *= 2;
        // maximize tgZ
        while (tgX * tgY * tgZ * 2 <= maxThreadsPerGroup && tgZ * 2 <= gridZ && tgZ * 2 <= maxTG.depth) tgZ *= 2;

        config.threadgroupSize = MTL::Size::Make(tgX, tgY, tgZ);
        return config;
    }

}


}
