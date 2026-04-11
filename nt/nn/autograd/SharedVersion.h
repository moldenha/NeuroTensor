#ifndef NT_NN_AUTOGRAD_SHARED_VERSION_H__
#define NT_NN_AUTOGRAD_SHARED_VERSION_H__

#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <atomic>
#include <cstdint>

namespace nt::grad::utility{

struct shared_version : public intrusive_ptr_target {
    std::atomic<uint64_t> version{0};
    uint64_t increment_version(){
        return this->version.fetch_add(1, std::memory_order_acq_rel);
    }
    uint64_t load() {
        return this->version.load(std::memory_order_acquire);
    }
};

} // nt::grad::utility::


#endif
