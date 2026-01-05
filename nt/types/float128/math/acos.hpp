#ifndef NT_MATH_FLOAT128_ACOS_HPP__
#define NT_MATH_FLOAT128_ACOS_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "asin.hpp"

// constexpr math functions
namespace nt::math{

NT_ALWAYS_INLINE float128_t acos(const float128_t& x) noexcept {
    float128_t a = ::nt::math::asin(x);
    if(a.get_bits().is_nan()) return a;
    return f128_constants::PI_2 - a;
}

}

#endif
