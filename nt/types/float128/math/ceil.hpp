#ifndef NT_MATH_FLOAT128_CEIL_HPP__
#define NT_MATH_FLOAT128_CEIL_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "../kmath/ceil.hpp"

// constexpr math functions
namespace nt::math{


NT_ALWAYS_INLINE float128_t ceil(const float128_t& x) noexcept {
    return kmath::ceil(x);
}


}

#endif
