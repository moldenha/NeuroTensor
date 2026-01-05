#ifndef NT_MATH_FLOAT128_ASIN_HPP__
#define NT_MATH_FLOAT128_ASIN_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "sqrt.hpp"
#include "atan.hpp"
#include "../kmath/abs.hpp"

// constexpr math functions
namespace nt::math{

inline float128_t asin(const float128_t& x) noexcept {
    float128_bits bits = x.get_bits();
    constexpr float128_t one = 1_f128;
    if(bits.is_nan()) return x;

    // |x| > 1 -> nan
    float128_t ax = kmath::abs(x);
    if(ax > 1) return float128_t::make_nan();
    // asin(+- 1) = +- pi/2
    if(x == float128_t(1)) return f128_constants::PI_2; 
    if(x == float128_t(-1)) return -f128_constants::PI_2; 

    // |x| <= 0.5 -> safe formula
    if(ax <= 0.5){
        float128_t t = ::nt::math::sqrt(one - (x * x));
        return ::nt::math::atan(x / t);
    }
    // |x| > 0.5 -> stable
    float128_t t = ::nt::math::sqrt((one - ax) / (one + ax));
    float128_t r = f128_constants::PI_2 - float128_t(2) * ::nt::math::atan(t);
    // float128_t r = f128_constants::PI_2 - ::nt::math::atan(t);
    return bits.sign() ? -r : r;

}

}

#endif
