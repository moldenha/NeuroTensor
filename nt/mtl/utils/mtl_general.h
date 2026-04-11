#ifndef NT_MTL_ABSTRACTION_MTL_GENERAL_H__
#define NT_MTL_ABSTRACTION_MTL_GENERAL_H__

#include "../../dtype/DType_enum.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <cstdint>
#include <atomic>
#include "../abstraction/mtl_buffer.h"
#include "../abstraction/mtl_context.h"

#ifdef NT_MTL_SUPPORTED

namespace nt::mtl{

static constexpr bool supported() { return true; }
static constexpr bool supported(DType dt){
    switch(dt){
        case DType::Float16:
            // defined as half
            return true;
        case DType::Float32:
            // defined as float
            return true;
        case DType::Float64:
           // double's are slow, and not supported everywhere, so excluded
           return false; 
        case DType::int8:
            return true;
        case DType::uint8:
            return true;
        case DType::int16:
            return true;
        case DType::uint16:
            return true;
        case DType::int32:
            return true;
        case DType::uint32:
            return true;
        case DType::int64:
            return true; // slow
        case DType::uint64:
            return true; // slow
        case DType::Complex64:
            // float2
            return true;
        case DType::Complex32:
            // half2
            return true;
        case DType::Bool:
            // bool
            return true;
        default:
            return false;
    }
}

// future iteration will have a synchronize for the nt::Tensor class
void synchronize(const intrusive_ptr<abs::MetalBuffer>& buf);
inline void synchronize() { abs::MetalContext::instance().flush(); }
inline void async(bool async_on) { abs::MetalContext::instance().async(async_on); }
}

#else

namespace nt::mtl{
static constexpr bool supported() { return false; }
static constexpr bool supported(DType) { return false; }
inline void synchronize(){;}
inline void async(bool async_on){;}
}

#endif

#endif
