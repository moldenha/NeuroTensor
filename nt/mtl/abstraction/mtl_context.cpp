#include "mtl_macros.h"

#include "mtl_context.h"
#include "mtl_pipeline.h"
#include "mtl_buffer.h"
#include "mtl_command.h"
#include "../utils.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <algorithm>
#include <functional>
#include <mutext>
#include <atomic>
#include <unordered_map>

namespace nt::mtl::abs {

MetalContext& MetalContext::instance(){
    static MetalContext ctx; // automatically destroyed at program end
    return ctx;
}

MetalContext::MetalContext()
    :pool_(NS::AutoreleasePool::alloc()->init()),
    device_(MTL::CreateSystemDefaultDevice()),
    library_(nullptr),
    queue_(nullptr),
    event_(nullptr),
    async_mode_(true)
{
    library_ = 
        this->device_->new_library(
            NS::String::string(METALLIB_PATH, NS::UTF8StringEncoding),
            nullptr
        );
    utils::throw_exception(library_ != nullptr,
            "Error: failed to internally load library");
    queue_ = 
        this->device_->newCommandQueue();
    event_ = this->device_->newSharedEvent();
    global_timeline.store(0, std::memory_order_release);
}

MetalContext::~MetalContext(){
    // clear all pipelines first
    std::unordered_map<std::string, intrusive_ptr<MetalPipeline>> empty_map;
    this->pipelines_.swap(empty_map);
    if(this->library_){
        this->library_->release();
        this->library_ = nullptr;
    }
    if(this->queue_){
        this->queue_->release();
        this->queue_ = nullptr;
    }
    if(this->event_){
        this->event_->release();
        this->event_ = nullptr;
    }
    if(this->pool_){
        this->pool_->drain();
        this->pool_ = nullptr;
    }
    if(this->device_){
        this->device_->release();
        this->device_ = nullptr;
    }
}

MTL::Device* MetalContext::device() noexcept {return device_;}
NS::AutoreleasePool* MetalContext::pool() noexcept { return pool_; }
MTL::Library* MetalContext::library() noexcept { return library_; }
MTL::CommandQueue* MetalContext::queue() noexcept { return queue_; }
intrusive_ptr<MetalPipeline> MetalContext::get_pipeline(
        const std::string& kernelName){
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        auto it = this->pipelines_.find(kernelName);
        if(it != this->pipelines_.end())
            return it->second;
    }
    // Create outside lock
    intrusive_ptr<MetalPipeline> pipeline = make_intrusive<MetalPipeline>(
        kernelName,
        this->library_,
        this->device_
    );

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = pipelines.emplace(kernelName, pipeline);
        if (!inserted)
            return it->second; // someone else won race
    }
    return pipeline;
}


template<typename T>
inline void helper_erase(std::vector<T>& vec, const T& val){
    auto it = std::remove(vec.begin(), vec.end(), val);
    vec.erase(it, vec.end());
}

// the point of this function
// is to ensure that the buffers being used are not being
// written to or read from in a way that could create the
// wrong output from the current function about to be ran
intrusive_ptr<MetalCommand> MetalContext::makeCommandBuffer(){
    // this before hand creates the command buffer
    // and encodes events into the reads and write
    // appropriately in order to waut for the events to be done
    MTL::CommandBuffer* cmd = queue_->commandBuffer();
    uint64_t ts = this->global_timeline.fetch_add(1, std::memory_order_acq_rel) + 1;
    // cmd->encodeSignalingEvent(this->event_, cur_timeline);
    return make_intrusive<MetalCommand>(cmd, ts);
}

// the above is called, then the encoder is given all the arguments
// and then the above is run at the end

void MetalContext::run_command(
        intrusive_ptr<MetalCommand> cmd,
        bool async
    ){
    cmd->cmd->encodeSignalingEvent(this->event_, cmd->timestamp);
    // Add it to the outstanding commands immediately (needed for safety)
    if (async_mode && async) {
        std::lock_guard<std::mutex> lock(mutex_);
        outstanding_commands_.emplace_back(cmd_i);
    }
    

    std::function<void(MTL::CommandBuffer*)> completion_event = 
        std::function<void(MTL::CommandBuffer*)>(
        [bool async_mode = this->async_mode,
        intrusive_ptr<MetalCommand> cmd = cmd,
        std::vector<intrusive_ptr<MetalCommand>>& cmds = this->outstanding_commands_,
        std::mutext& mutex = this->mutex_]
        (MTL::CommandBuffer* buf){
        if(async_mode){
            std::lock_guard<std::mutex> lock(mutex);
            auto it = std::find(cmds.begin(), cmds.end(), cmd);
            if(it != cmds.end())
                cmds.erase(it);
        }
        if(buf && cmd && cmd->cmd){
            cmd->cmd->release();
            cmd->cmd = nullptr;
            cmd->release_args();
        }else if(cmd){
            cmd->release_args();
        }
    });
    // Add a completion handler to automatically remove it when done
    if(aync_mode && async)
        cmd->cmd->addCompletedHandler(completion_event);


    cmd->cmd->commit();

    // Synchronous mode waits
    if(!async_mode || !async)
        cmd->cmd->waitUntilCompleted();
}

void MetalContext::flush(){
    for(auto cmd : this->outstanding_commands_){
        cmd->flush();
    }
    this->outstanding_commands_.clear();
}

void MetalContext::async(bool async_on){
    if(async_on){
        this->async_mode_ = true;
        return;
    }
    this->flush();
    this->async_mode_ = false;
}

}
