#if !defined(NT_MTL_ABSTRACTION_MTL_ENCODE_TENSOR_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_ENCODE_TENSOR_H__

#include "../mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>


#include "encoding_utils.h"
#include "../mtl_arg_encoder.h"
#include "../mtl_buffer.h"
#include "../mtl_command.h"
#include "../mtl_pipeline.h"

namespace nt::mtl::abs::encoder_tensor_details {



struct NtGPUTensorCPU {
    void* data;
    void* indices;
    int layout;
    int ndim;
    int64_t numel;
    int64_t offset;
    int64_t sizes[8];
    int64_t strides[8];
};

struct DispatchParams{
    int64_t base;
};

inline void encodeNtGPUTensor(
    intrusive_ptr<MetalArgEncoder> args,
    intrusive_ptr<MetalCommand& cmd, 
    NS::Array* bindings,
    const int64_t& binding_index,
    intrusive_ptr<MetalBuffer> data,
    intrusive_ptr<MetalBuffer> indices,
    int layout,
    int ndim,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t numel,
    int64_t offset){

    // Patch device pointers
    args->set_buffer(data, 0, 0);
    args->set_buffer(indices, 0, 1);

    // sync the data
    ::nt::mtl::abs::utils::handle_buffer_sync(cmd, data, bindings, binding_index);
    // NOTE: still haven't decided if this is needed 
    /* if(indices != nullptr) ::nt::mtl::abs::utils::handle_buffer_sync(cmd, indices, bindings, binding_index); */

    // Scalars
    args->set_argument<int>(2) = layout;
    args->set_argument<int>(3) = ndim;
    args->set_argument<int64_t>(4) = numel;
    args->set_argument<int64_t>(5) = offset;

    auto* view = reinterpret_cast<NtGPUTensorCPU*>(args->contents());
    // Arrays
    int64_t* sizesPtr =
        reinterpret_cast<int64_t*>(view->sizes);

    int64_t* stridesPtr =
        reinterpret_cast<int64_t*>(view->strides);

    for (int i = 0; i < 8; ++i)
    {
        sizesPtr[i]   = (sizes && i < ndim)   ? sizes[i]   : 1;
        stridesPtr[i] = (strides && i < ndim) ? strides[i] : 0;
    }

    // encoder->setBuffer(argBuffer, 0, bufferIndex);
}

inline MTL::Binding* find_index(const int64_t& index,
                        NS::Array* bindings){
    MTL::Binding* binding = bindings->object<MTL::Binding>(index);
    if(index == binding->index())
        return binding;
    NS::UInteger size = bindings->count();
    for(NS::UInteger i =0; i < size; ++i){
        if(MTL::Binding* bind = bindings->object<MTL::Binding>(i)){
            if(bind->index() == index) return bind;
        }
    }
    return nullptr;
}

inline void EncodeMTLBucket(
                        intrusive_ptr<MetalArgEncoder> args,
                        intrusive_ptr<BucketMTL> bucket,
                        const int64_t& binding_index,
                        NS::Array* bindings,
                        intrusive_ptr<MetalCommand> cmd,
                        int64_t concat_index = 0){
    uint32_t index = bucket->storage.index();
    if(index == 3){
        // concatenated
        // Currently: this function only handles one concatenated tensor at a time
        // There is going to be a function to abstract away the concatenation so that the command buffer will
        //  be able to just run the kernel multiple times with multiple concatenations
        const auto& s = std::get<3>(bucket->storage);
        intrusive_ptr<BucketMTL> mtl_bucket = make_intrusive<BucketMTL>(s.buffers[concat_index], 
                                                                        s.devices->get(concat_index), 
                                                                        bucket->dtype()
        );
        EncoderMTLBucket(args, mtl_bucket, binding_index, bindings, concat_index);
    }
    MTL::Binding* binding = find_index(index, bindings);
        MTL::Binding* binding = find_index(index, bindings);
    utils::throw_exception(parameter != nullptr,
            "Error, encoding buffer that does not exist in the kernel");
    utils::THROW_EXCEPTION(
            binding->type() == MTL::BindingTypeBuffer,
            "Error, got wrong binding type from kernel, expected buffer");
    MTL::BufferBinding* buf_binding = reinterpret_cast<MTL::BufferBinding*>(binding);
    MTL::PointerType* ptype = buf_binding->bufferPointerType();
    utils::throw_exception(ptype != nullptr,
            "Error, expected buffer to have a pointer type when encoding a device");
    MTL::BindingAccess access = ptype->access();
    
    intrusive_ptr<MetalBuffer> data_buffer = bucket->get_buffer();
    if(access == MTL::BindingAccessReadOnly){
        uint64_t wait = data_buffer->last_write();
        if(wait > 0 && wait < timestamp){
            cmd->cmd->encodeWaitForEvent(
                event,
                wait
            );
        }
        data_buffer->read_timeline.store(timestamp, std::memory_order_release);
    }else{
        // read-write or write access
        uint64_t wait = data_buffer->greatest_timeline();
        if(wait > 0 && wait < timestamp){
            cmd->cmd->encodeWaitForEvent(
                    event,
                    wait
            );
        }
        data_buffer->write_timeline.store(timestamp, std::memory_order_release);
    }


    if(index == 0){
        // contiguous
        const auto& s = std::get<0>(bucket->storage);
        encodeNtGPUTensor(
                /*args = */ args,
                /*cmd = */ cmd,
                /*bindings = */bindings,
                /*binding_index = */binding_index,
                /*dataBuffer = */data_buffer,
                /*indicesBuffer = */nullptr,
                /*layout = */0, // contiguous
                /*ndim = */0,
                /*sizes = */nullptr,
                /*strides = */nullptr,
                /*numel = */s.numel,
                /*offset = */s.offset
        ); 
    }
    else if(index == 1){
        // affine
        const auto& s = std::get<1>(bucket->storage);
        encodeNtGPUTensor(
                /*args = */ args,
                /*cmd = */ cmd,
                /*bindings = */bindings,
                /*binding_index = */binding_index,
                /*dataBuffer = */data_buffer,
                /*indicesBuffer = */nullptr,
                /*layout = */1, // affine
                /*ndim = */s.ndim,
                /*sizes = */s.sizes(),
                /*strides = */s.strides(),
                /*numel = */s.numel(),
                /*offset = */s.offset
        );
    }
    else if(index == 2){
        // strided
        const auto& s = std::get<2>(bucket->storage);
        encodeNtGPUTensor(
                /*args = */ args,
                /*cmd = */ cmd,
                /*bindings = */bindings,
                /*binding_index = */binding_index,
                /*indicesBuffer = */s.indexes->get_buffer(),
                /*dataBuffer = */data_buffer,
                /*layout = */2, // strided
                /*ndim = */0,
                /*sizes = */nullptr,
                /*strides = */nullptr,
                /*numel = */s.nnz,
                /*offset = */s.idx_offset
        );
    }
}


}
