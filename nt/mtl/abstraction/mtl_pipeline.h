#if !defined(NT_MTL_ABSTRACTION_MTL_PIPELINE_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_PIPELINE_H__

#include "mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
// #include "../../dtype/DType_enum.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <string>
#include <cstdint>
// #include "../../memory/meta_allocator.h"
// #include <unordered_map>
// #include <mutex>
// #include <exception>
// #include <vector>
// #include <atomic>

namespace nt::mtl::abs{

class MetalPipeline : intrusive_ptr_target {
    MTL::Function* kernelFunc_;
    MTL::ComputePipelineDescriptor* desc_;
    MTL::ComputePipelineState* pipeline_;
    MTL::ComputePipelineReflection* reflection_;
    public:
        MetalPipeline() = delete;
        MetalPipeline(const std::string& kernelName, 
                MTL::Library* library, 
                MTL::Device* device)
            :kernelFunc_(library->newFunction(NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding))),
            desc_(nullptr),
            pipeline_(nullptr),
            reflection_(nullptr)
        {
            if (!kernelFunc_)
                throw std::runtime_error("Failed to load kernel: " + kernelName);
            desc_ = MTL::ComputePipelineDescriptor::alloc()->init();
            desc_->setComputeFunction(this->kernelFunc_);
            desc_->setSupportIndirectCommandBuffers(true); // allow argument buffers
            NS::Error* error = nullptr;
            pipeline_ = device->newComputePipelineState(
                    this->desc_, 
                    MTL::PipelineOptionArgumentInfo,
                    &reflection_,
                    &error
            );
            if (!pipeline_) {
                std::string msg = error
                    ? error->localizedDescription()->utf8String()
                    : "Unknown Metal error";
                throw std::runtime_error("Pipeline creation failed: " + msg);
            }
            if(reflection_)
                reflection_->retain();

        }
        MetalPipeline(const Pipeline&) = delete;
        MetalPipeline& operator=(const Pipeline&) = delete;
        ~MetalPipeline(){
            if(desc_){
                desc_->release();
                desc_ = nullptr;
            }
            if (pipeline_){
                pipeline_->release();
                pipeline_ = nullptr;
            }
            if(kernelFunc_){
                kernelFunc_->release();
                kernelFunc_ = nullptr;
            }
            if(reflection_){
                reflection_->release();
                reflection_ = nullptr;
            }
        }
        inline MTL::ComputePipelineState* pipeline() noexcept {return pipeline_;}
        inline MTL::ComputePipelineReflection* reflection() noexcept {return reflection_;}
        inline MTL::Function* kernelFunc() noexcept {return kernelFunc_;}
};




}

#endif
