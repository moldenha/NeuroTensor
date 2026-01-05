#ifndef NT_MATH_FLOAT128_ABS_HPP__
#define NT_MATH_FLOAT128_ABS_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "../kmath/abs.hpp"

// constexpr math functions
namespace nt::math{


NT_ALWAYS_INLINE float128_t abs(const float128_t& x) noexcept {
    return kmath::abs(x);
}


}

#endif
