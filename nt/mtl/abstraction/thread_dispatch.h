#if !defined(NT_MTL_ABSTRACTION_THREAD_DISPATCH_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_THREAD_DISPATCH_H__

#include "mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <cstdint>
// #include "../../dtype/DType_enum.h"
// #include "../../intrusive_ptr/intrusive_ptr.hpp"
// #include "../../memory/meta_allocator.h"
// #include <unordered_map>
// #include <mutex>
// #include <exception>
// #include <vector>
// #include <atomic>

namespace nt::mtl::abs{

struct ThreadDispatchConfig {
    MTL::Size gridSize;
    MTL::Size threadgroupSize;
};

ThreadDispatchConfig computeThreadDispatchConfig(int64_t N);


}

#endif
