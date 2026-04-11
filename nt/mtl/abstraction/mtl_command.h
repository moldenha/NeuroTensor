#if !defined(NT_MTL_ABSTRACTION_MTL_COMMAND_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_COMMAND_H__

// the reason for this struct is for 2 reasons:
//  - a holding tracker for memory saftey
//  - this struct will be the only thing that directly has a MTL::Buffer*
//  - but the release and everything will be managed by different objects
//  - so that a hash can be made of memory buffers

#include "mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <vector>
// #include "../../dtype/DType_enum.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <cstdint>
#include "../../memory/meta_allocator.h"
// #include <unordered_map>
// #include <mutex>
#include <exception>
// #include <vector>
#include <atomic>
#include "mtl_arg_encoder.h"

namespace nt::mtl::utils{

struct MetalCommand : intrusive_ptr_target{
    MTL::CommandBuffer* cmd;
    std::vector<intrusive_ptr<MetalArgEncoder>> arg_encoders;
    const uint64_t timestamp;
    MetalCommand(MTL::CommandBuffer* cmd_, uint64_t timestamp_)
        :cmd(cmd_), timestamp(timestamp_)
    {}
    MetalCommand() = delete;
    inline void flush(){
        if(cmd){
            cmd->waitUntilCompleted();
            // cmd->release();
            cmd = nullptr;
        }
    }
    inline void release_args(){
        for(auto& encoder : arg_encoders)
            encoder.reset(nullptr);
        this->arg_encoders.clear();
    }

};





}

#endif
