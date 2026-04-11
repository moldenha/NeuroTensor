#include "encoding_utils.h"

namespace nt::mtl::abs::utils{

MTL::Binding* find_index(const int64_t& index,
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

// Will have to make something similar to this when doing the tensor encoding for the bucket (not super different, jut accessing the different buffers)
void handle_buffer_sync(intrusive_ptr<MetalCommand>& cmd, intrusive_ptr<MetalBuffer> buf, 
        NS::Array* bindings, const int64_t& index){
    MTL::Binding* binding = find_index(index, bindings);
    utils::throw_exception(binding != nullptr,
            "Error, encoding buffer that does not exist in the kernel");
    utils::THROW_EXCEPTION(
            binding->type() == MTL::BindingTypeBuffer,
            "Error, got the wrong binding type from the kernel, expected a buffer");
    MTL::BufferBinding* buf_binding = reinterpret_cast<MTL::BufferBinding*>(binding);
    // this would pretty much be what has to change here for encoding the nt gpu tensor structure:
    MTL::PointerType* ptype = buf_binding->bufferPointerType();
    utils::THROW_EXCEPTION(ptype != nullptr,
                            "Error, expected the buffer argument to have a pointer type when encoding a device");
    MTL::BindingAccess access = ptype->access();
    uint64_t timestamp = cmd->timestamp;
    MetalContext& ctx = MetalContext::instance();
    MTL::SharedEvent* event = ctx.event();
    if(access == MTL::BindingAccessReadOnly){
        uint64_t wait = arg.val->last_write();
        if(wait > 0 && wait < timestamp){
            cmd->cmd->encodeWaitForEvent(
                event,
                wait
            );
        }
        arg.val->read_timeline.store(timestamp, std::memory_order_release);
    }else{
        // read-write or write access
        uint64_t wait = arg.val->greatest_timeline();
        if(wait > 0 && wait < timestamp){
            cmd->cmd->encodeWaitForEvent(
                    event,
                    wait
            );
        }
        arg.val->write_timeline.store(timestamp, std::memory_order_release);
    }


}

}
