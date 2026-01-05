#ifndef NT_MATH_FLOAT128_FMOD_HPP__
#define NT_MATH_FLOAT128_FMOD_HPP__

#include "../float128_impl.h"
#include "../../../utils/always_inline_macro.h"
#include "../kmath/trunc.hpp"

namespace nt::math{


// could be constexpr
inline float128_t fmod(float128_t x, float128_t y) noexcept {
    float128_bits xb = x.get_bits();
    float128_bits yb = y.get_bits();
    // nan cases
    if (xb.is_nan() || yb.is_nan())
        return float128_t::make_nan();

    // division by zero
    if (yb.is_zero())
        return float128_t::make_nan();

    // infinite x
    if (xb.is_inf())
        return float128_t::make_nan();

    // infinite y -> result is x
    if (yb.is_inf())
        return x;

    // normal case: x - trunc(x/y) * y
    float128_t q = kmath::trunc(x / y);
    float128_t r = x - q * y;

    // Ensure signed zero matches x
    if(r.get_bits().is_zero())
        return float128_t::make_zero(xb.sign());
    return r;
}



}

#endif
