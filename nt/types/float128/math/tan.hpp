#ifndef NT_MATH_FLOAT128_TAN_HPP__
#define NT_MATH_FLOAT128_TAN_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "sin.hpp"
#include "cos.hpp"

// constexpr math functions
namespace nt::math{

NT_ALWAYS_INLINE float128_t tan(const float128_t& x) noexcept {
    return ::nt::math::sin(x) / ::nt::math::cos(x);
}

}

#endif
