#if !defined(NT_MTL_ABSTRACTION_MTL_BUFFER_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_BUFFER_H__

#include "mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
// #include "../../dtype/DType_enum.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <cstdint>
#include "../../memory/meta_allocator.h"
// #include <unordered_map>
#include <mutex>
// #include <exception>
// #include <vector>
#include <atomic>

namespace nt::mtl::abs{


struct MetalBuffer {
    MTL::Buffer* buffer = nullptr;
    size_t typeBytes;
    std::atomic<uint64_t> write_timeline{0};
    std::atomic<uint64_t> read_timeline{0};

    MetalBuffer(MTL::Device* device,
                int64_t size, size_t type_size,
                MTL::ResourceOptions opts)
    :typeBytes(type_size)
    {
        buffer = device->newBuffer(size, opts);

        write_timeline.store(0, std::memory_order_release);
        read_timeline.store(0, std::memory_order_release);
    }
    ~MetalBuffer(); 
    inline uint64_t last_write() const noexcept {
        return this->write_timeline.load(std::memory_order_acquire);
    }
    inline uint64_t last_read() const noexcept {
        return this->read_timeline.load(std::memory_order_acquire);
    }
    inline uint64_t greatest_timeline() const noexcept {
        uint64_t write = last_write();
        uint64_t read = last_read();
        return write < read ? read : write;
    }
    void* contents() noexcept {return buffer->contents();}
};

struct MetalBufferView{
    intrusive_ptr<MetalBuffer> buffer;
    int64_t offsetBytes, numelBytes, idxOffset;
    uint8_t ndim;
    intrusive_ptr<intrusive_tracked_list_sub<int64_t, false>> sizes, strides;
    intrusive_ptr<DeviceMTLPrivate> indexes;
};

}

#endif
