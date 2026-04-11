#include "mtl_macros.h"
#include "mtl_buffer.h"
#include "mtl_context.h"
#include <mutex>
#include <conditional_variable>
#include <cstdint>

namespace nt::mtl::abs {

// these should only really be used in specific cases
// void MetalBuffer::clear_readers(){
//     std::lock_guard<std::mutex> lock(mtx);
//     for(auto& reader : readers){
//         reader->flush();
//         reader.reset(nullptr);
//     }
//     readers.clear();
// }

// void MetalBuffer::clear_writers(){
//     std::lock_guard<std::mutex> lock(mtx);
//     for(auto& writer : writers){
//         writer->flush();
//         writer.reset(nullptr);
//     }
//     writers.clear();
// }

// void MetalBuffers::flush(){
//     std::lock_guard<std::mutex> lock(mtx);
//     for(auto& reader : readers){
//         reader->flush();
//         reader.reset(nullptr);
//     }
//     readers.clear();
//     for(auto& writer : writers){
//         writer->flush();
//         writer.reset(nullptr);
//     }
//     writers.clear();
// }

// void MetalBuffer::release(){
//     if(buffer){
//         buffer->release();
//         buffer = nullptr;
//     }
//     this->flush();
// }

void wait_for_event(MTL::SharedEvent* event, uint64_t value)
{
    std::mutex m;
    std::condition_variable cv;
    bool done = false;

    auto listener = MTL::SharedEventListener::alloc()->init();

    event->notifyListener(listener, value,
        ^(MTL::SharedEvent*, uint64_t) {
            std::lock_guard<std::mutex> lock(m);
            done = true;
            cv.notify_one();
        });

    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&]{ return done; });

    listener->release();
}

MetalBuffer::~MetalBuffer(){
    wait_for_event(MetalContext::instance().event(), this->greatest_timeline());
    if(buffer){
        buffer->release();
        buffer = nullptr;
    }
}

}
