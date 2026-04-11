#include "../abstraction.h"
#include "../../device.h"
#include "reduce_and.h"
#include "../utils/mtl_general.h"
#include <cstring>
#include <limits>

namespace nt::mtl{


// this isn't going to use as much abstraction because it is pretty custom
// and this ping-pong method allows for faster results

// this version is going to happen when N is less than the max size of (uint32_t - 256)
template<typename T>
bool reduce_and_uint32_low_size(const intrusive_ptr<abs::MetalBuffer>& flags, 
                                const uint32_t& N, int64_t offset, std::string type){
    
    abs::MetalContBucketMTL& out_bucketext& ctx = abs::MetalContext.instance();
    intrusive_ptr<abs::Pipeline> reduceAndPSO = ctx.get_pipeline("reduce_and_kernel_" + type);
    MTL::CommandBuffer* cmd = ctx.queue()->commandBuffer();
    MTL::Device* = ctx.device();
    MTL::Buffer* in  = flags->buffer;
    MTL::Buffer* out = nullptr;

    uint32_t currentN = N;
    bool first_pass = true;
    while(currentN > 1){
        uint32_t nextN = (currentN + 255) / 256;
        out = device->newBuffer(
            sizeof(T) * nextN,
            MTL::ResourceStorageModePrivate
        );
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(reduceAndPSO->pipeline);
        enc->setBuffer(in, (first_pass && offset > 0) ? offset : 0, 0);
        enc->setBuffer(out, 0, 1);
        enc->setBytes(&currentN, sizeof(uint32_t), 2);
        enc->dispatchThreads(
            MTL::Size(nextN * 256, 1, 1),
            MTL::Size(256, 1, 1)
        );
        enc->endEncoding();
        in = out;
        currentN = nextN;
        first_pass = false;
    }
    cmd->commit();
    cmd->waitUntilCompleted();

    uint32_t result;
    MTL::Buffer* readback = device->newBuffer(
        sizeof(T),
        MTL::ResourceStorageModeShared
    );

    auto* blit = cmd->blitCommandEncoder();
    blit->copyFromBuffer(in, 0, readback, 0, sizeof(uint32_t));
    blit->endEncoding();

    cmd->commit();
    cmd->waitUntilCompleted();

    result = *(T*)readback->contents();
    if constexpr (std::is_same_v<T, bool>){
        return result;
    }else{
        return (result == 1); // all true
    }
}

template<typename T>
bool reduce_and_high_size(const intrusive_ptr<abs::MetalBuffer>& flags, 
                                int64_t N, int64_t offset, std::string type){
    
    uint32_t max_size = std::numeric_limits<uint32_t>::max() - 256;
    if(N - int64_t(max_size) < 0){
        return reduce_and_uint32_low_size(flags, static_cast<uint32_t>(N), offset);
    }
    abs::MetalContBucketMTL& out_bucketext& ctx = abs::MetalContext.instance();
    intrusive_ptr<abs::Pipeline> reduceAndPSO = ctx.get_pipeline("reduce_and_kernel_" + type);
    MTL::CommandBuffer* cmd = ctx.queue()->commandBuffer();
    MTL::Device* = ctx.device();
    MTL::Buffer* in  = flags->buffer;
    MTL::Buffer* out = nullptr;


    while(N > 0){
        uint32_t currentN = std::min(max_size, N);
        offset += static_cast<int64_t>(max_size);
        N -= static_cast<int64_t>(max_size);
        if(N < 0)
            offset += N;
        bool first_pass = true;
        while(currentN > 1){
            uint32_t nextN = (currentN + 255) / 256;
            out = device->newBuffer(
                sizeof(T) * nextN,
                MTL::ResourceStorageModePrivate
            );
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(reduceAndPSO->pipeline);
            enc->setBuffer(in, (first_pass && offset > 0) ? offset : 0, 0);
            enc->setBuffer(out, 0, 1);
            enc->setBytes(&currentN, sizeof(uint32_t), 2);
            enc->dispatchThreads(
                MTL::Size(nextN * 256, 1, 1),
                MTL::Size(256, 1, 1)
            );
            enc->endEncoding();
            in = out;
            currentN = nextN;
            first_pass = false;
        }
    }
    cmd->commit();
    cmd->waitUntilCompleted();

    uint32_t result;
    MTL::Buffer* readback = device->newBuffer(
        sizeof(uint32_t),
        MTL::ResourceStorageModeShared
    );

    auto* blit = cmd->blitCommandEncoder();
    blit->copyFromBuffer(in, 0, readback, 0, sizeof(uint32_t));
    blit->endEncoding();

    cmd->commit();
    cmd->waitUntilCompleted();

    result = *(T*)readback->contents();
    if constexpr (std::is_same_v<T, bool>){
        return result;
    }else{
        return (result == 1); // all true
    }
}

template<typename T>
void reduce_and(intrusive_ptr<abs::MetalBuffer> dev, int64_t size, int64_t offset,
                        std::string type){
    utils::throw_exception(size >= 0 && offset >= 0,
            "Error, got wrong size ($) and offset ($) for reduce_and", size, offset);

    // make sure that the buffer is actually done being written to
    ::nt::mtl::synchronize(dev);
    uint32_t max_size = std::numeric_limits<uint32_t>::max() - 256;
    if(static_cast<int64_t>(max_size) > size){
        return reduce_and_low_size<T>(dev, static_cast<uint32_t>(size), offset, std::move(type));
    }
    return reduce_and_high_size<T>(dev, size, offset, std::move(type));
}

void reduce_and_uint32(intrusive_ptr<DeviceMTLPrivate> dev, int64_t size, int64_t offset){
    reduce_and<uint32_t>(dev->get_buffer(), size, start, offset, "uint");
}
void reduce_and_uint32(intrusive_ptr<DeviceMTLShared> dev, int64_t size, int64_t offset){
    reduce_and<uint32_t>(dev->get_buffer(), size, start, offset, "uint");
}


}
