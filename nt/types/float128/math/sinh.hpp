#ifndef NT_MATH_FLOAT128_SINH_HPP__
#define NT_MATH_FLOAT128_SINH_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "exp.hpp"

// constexpr math functions
namespace nt::math{

NT_ALWAYS_INLINE float128_t sinh(const float128_t& x) noexcept {
    float128_t e = ::nt::math::exp(x);
    return (e - (float128_t(1) / e)) / 2;
    // return (::nt::math::exp(x) - ::nt::math::exp(-x)) / 2;
}

}

#endif
