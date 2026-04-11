#include "../abstraction.h"
#include "../../device.h"
#include "iota.h"

namespace nt::mtl{

void iota_int64(intrusive_ptr<abs::MetalBuffer> dev, int64_t size, int64_t start, int64_t offset){
    utils::throw_exception(size >= 0 && offset >= 0,
            "Error, got wrong size ($) and offset ($) for iota", size, offset);
    abs::MetalContBucketMTL& out_bucketext& ctx = abs::MetalContext::instance();
    
    intrusive_ptr<abs::Pipeline> pipeline_ = ctx.get_pipeline("iota_contiguous_kernel_long");
    // MTL::CommandQueue* queue = ctx.queue();
    MTL::CommandBuffer* commandBuffer = commandBuffer_->cmd;

    //compute pass
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(dev->buffer, offset * sizeof(int64_t), 0);
    encoder->setBytes(&start, sizeof(int64_t), 1);

    utils::ThreadDispatchConfig config = utils::computeThreadDispatchConfig(size);
    encoder->setByes(&config.gridSize, sizeof(MTL::Size), 2);
    encoder->dispatchThreads(config.gridSize, config.threadgroupSize);
    encoder->endEncoding();
    
    ctx.run_command(commandBuffer_);
}

void iota_int64(intrusive_ptr<DeviceMTLPrivate> dev, int64_t size, int64_t start, int64_t offset){
    iota_int64(dev->get_buffer(), size, start, offset);
}
void iota_int64(intrusive_ptr<DeviceMTLShared> dev, int64_t size, int64_t start, int64_t offset){
    iota_int64(dev->get_buffer(), size, start, offset);
}


}
