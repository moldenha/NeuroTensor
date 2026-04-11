#if !defined(NT_MTL_ABSTRACTION_MTL_CONTEXT_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_CONTEXT_H__

#include "mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
// #include "../../dtype/DType_enum.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <cstdint>
#include "../../memory/meta_allocator.h"
#include <unordered_map>
#include <mutex>
#include <exception>
#include <vector>
#include <atomic>
#include <span>
#include "mtl_buffer.h"
#include "mtl_command.h"
#include "mtl_pipeline.h"

namespace nt::mtl::abs{

class MetalContext{
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    NS::AutoreleasePool* pool_ = nullptr;
    MTL::Device* device_ = nullptr;
    MTL::Library* library_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    std::atomic<uint64_t> global_timeline{0};
    MTL::SharedEvent* event_ = nullptr;
    std::unordered_map<std::string,
                intrusive_ptr<MetalPipeline>
            > pipelines_;
    std::mutex mutex_;
    // This holds a list of command buffers that have yet to be
    //      waited to be completed
    // This is nice for async features that can be turned on and off
    // 
    // This map is so that it can be easily determined if a buffer is being
    //  modified or needed to be modified


    std::vector<intrusive_ptr<MetalCommand>> outstanding_commands_;
    bool async_mode_;



    public:
        static MetalContext& instance();

        MTL::Device* device() noexcept;
        NS::AutoreleasePool* pool() noexcept;
        MTL::Library* library() noexcept;
        MTL::CommandQueue* queue() noexcept;
        MTL::SharedEvent* event() noexcept;
        intrusive_ptr<MetalPipeline> get_pipeline(const std::string& kernelName);
        // this function should be run every time a command
        // buffer needs to be run
        intrusive_ptr<MetalCommand> MetalContext::makeCommandBuffer();
        void run_command(
            intrusive_ptr<MetalCommand> cmd,
            bool async = true);
        void flush();
        // this toggles on/off async
        // so basically if it is on
        //  - on -> cpu will not wait for gpu function to finish
        //      it will just continue and can add more workloads to gpu
        //  - off -> cpu will wait for gpu function to finish
        void async(bool asyn_on);
};

}

#endif


