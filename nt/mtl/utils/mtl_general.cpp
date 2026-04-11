#include "mtl_macros.h"
#include "mtl_general.h"
#include "mtl_context.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <mutex>
#include <conditional_variable>

namespace nt::mtl{

void wait_for_shared_event(MTL::SharedEvent* event, uint64_t value)
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

void synchronize(const intrusive_ptr<abs::MetalBuffer>& buf){
    wait_for_shared_event(utils::MetalContext::instance().event(), buf->greatest_timeline());
}

}
