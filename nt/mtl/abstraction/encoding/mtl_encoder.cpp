#include "mtl_macros.h"
#include "mtl_encoder.h"
#include "mtl_command.h"
#include "mtl_pipeline.h"
#include "mtl_context.h"
#include "mtl_buffer.h"
#include "../../utils/utils.h"
#include <atomic>
#include <mutex>

namespace nt::mtl::abs{

namespace details{

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

void setEncoderBuffer(int64_t& index,
                        MTL::ComputeCommandEncoder* encoder,
                        intrusive_ptr<MetalCommand>& cmd,
                        NS::Array* bindings,
                        EncoderVariable<intrusive_ptr<MetalBuffer>> arg){
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
    encoder->setBuffer(
        arg.val->buffer,
        arg.val.offset,
        index
    );
    ++index;
}

}

}
