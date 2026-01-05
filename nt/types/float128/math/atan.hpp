#ifndef NT_MATH_FLOAT128_ATAN_HPP__
#define NT_MATH_FLOAT128_ATAN_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../b256.h"
#include "../from_string.hpp"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "../kmath/copysign.hpp"

// constexpr math functions
namespace nt::math{

namespace details{

NT_ALWAYS_INLINE float128_t atan_kernel(float128_t x) noexcept {
    // assumes |x| <= tan(pi/8)
    float128_t x2 = x * x;

    // x - x^3/3 + x^5/5 - x^7/7
    float128_t r = x;
    float128_t term = x * x2;

    r -= term / float128_t(3);
    term *= x2;
    r += term / float128_t(5);
    term *= x2;
    r -= term / float128_t(7);

    return r;
}

}


inline float128_t atan(const float128_t& x_) noexcept {
    float128_bits bits = x_.get_bits();

    if(bits.is_nan()) return x_;
    if(bits.is_zero()) return x_;
    if(bits.is_inf()) return kmath::copysign(f128_constants::PI_2, x_); 

    bool neg = bits.sign();
    float128_t x = neg ? -x_ : x_;

    float128_t result;

    if(x > float128_t(1)) {
        // atan(x) = pi/2 - atan(1/x)
        float128_t t = float128_t(1) / x;
        if(t > f128_constants::TAN_PI_8) {
            float128_t u = (t - float128_t(1)) / (t + float128_t(1));
            result = f128_constants::PI_2 -
                     (f128_constants::PI_4 + details::atan_kernel(u));
        } else {
            result = f128_constants::PI_2 - details::atan_kernel(t);
        }
    }
    else if(x > f128_constants::TAN_PI_8) {
        // atan(x) = pi/4 + atan((x-1)/(x+1))
        float128_t t = (x - float128_t(1)) / (x + float128_t(1));
        result = f128_constants::PI_4 + details::atan_kernel(t);
    }
    else {
        result = details::atan_kernel(x);
    }

    return neg ? -result : result;
}



}

#endif
