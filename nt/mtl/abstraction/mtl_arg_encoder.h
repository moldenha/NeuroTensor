#if !defined(NT_MTL_ABSTRACTION_MTL_ARG_ENCODER_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_ARG_ENCODER_H__

#include "mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
// #include "../../dtype/DType_enum.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <string>
#include <cstdint>
#include "mtl_buffer.h"
#include "mtl_pipeline.h"
// #include "../../memory/meta_allocator.h"
// #include <unordered_map>
// #include <mutex>
// #include <exception>
// #include <vector>
// #include <atomic>

namespace nt::mtl::abs{

class MetalArgEncoder : intrusive_ptr_target {
    MTL::ArgumentEncoder* encoder_;
    MTL::Buffer* arg_buffer_; // not an intrusive MetalBuffer because the extra overhead and tracking is not needed
    
    public:
        MetalArgEncoder() = delete;
        MetalArgEncoder(MTL::Device* device, intrusive_ptr<MetalPipeline> pipeline, uint32_t arg_index)
            :encoder_(pipeline->kernelFunc()->newArgumentEncoder(arg_index)),
            arg_buffer_(nullptr)
        {
            arg_buffer_ = device->newBuffer(encoder_->encodedLength(),MTL::ResourceStorageModeShared);
            encoder_->setArgumentBuffer(arg_buffer_, 0);
        }
    
        ~MetalArgEncoder(){
            if(encoder_){
                encoder_->release();
                encoder_ = nullptr;
            }
            if(arg_buffer_){
                arg_buffer_->release();
                arg_buffer_ = nullptr;
            }
        }
        template<typename T>
        inline T& set_argument(uint32_t index, const T& n_val) noexcept {
            T& val = *reinterpret_cast<T*>(this->encoder_->constantData(index));
            val = n_val;
            return val;
        }
        
        template<typename T>
        inline T& set_argument(uint32_t index) noexcept { 
            return *reinterpret_cast<T*>(this->encoder_->constantData(index));
        }

        inline void set_buffer(intrusive_ptr<MetalBuffer> buffer, int64_t offset, uint32_t index) noexcept {
            if(buffer == nullptr){
                this->encoder_->setBuffer(nullptr, 0, index);
            }else{
                this->encoder_->setBuffer(buffer->buffer_, offset, index);
            }
        }

        inline void* contents() noexcept { return this->encoder_->contents(); }
        inline void finish_encoding() noexcept {
            this->encoder_->release();
            this->encoder_ = nullptr;
        }
        inline MTL::Buffer* get_buffer() noexcept { return this->arg_buffer(); }
        
        
};

}

#endif
