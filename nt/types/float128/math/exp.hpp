#ifndef NT_MATH_FLOAT128_EXP_HPP__
#define NT_MATH_FLOAT128_EXP_HPP__

#include "../float128_impl.h"
#include "../b256.h"
#include "../from_string.hpp"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "../kmath/round.hpp"
#include "../kmath/ldexp.hpp"
#include "../kmath/abs.hpp"

// constexpr math functions
namespace nt::math{


inline float128_t exp(const float128_t& x_) noexcept {
    float128_bits bits = x_.get_bits();
    bool negative = false;
    if(bits.is_nan()) return x_;
    if(bits.is_zero()) return float128_t(1);
    float128_t x = x_;
    if(bits.sign()){
        if(x < f128_constants::EXP_MIN) return float128_t::make_zero();
        bits = bits.positive_sign_bit();
        negative = true;
        x = kmath::abs(x);
    }
    if(bits.is_inf()) return x;
    
    // Overflow / underflow clamps
    if (x > f128_constants::EXP_MAX) return float128_t::make_inf(false);

    float128_t y = x * f128_constants::INV_LN2;
    float128_t k = kmath::round(y);
    float128_t r = x - float128_t(k) * f128_constants::LN2;

    float128_t p = float128_t(1);
    float128_t term = float128_t(1);

    term *= r; p += term;
    term *= r / float128_t(2); p += term;
    term *= r / float128_t(3); p += term;
    term *= r / float128_t(4); p += term;
    term *= r / float128_t(5); p += term;
    term *= r / float128_t(6); p += term;
    term *= r / float128_t(7); p += term;

    return negative ? float128_t(1) / (p * kmath::ldexp(float128_t(1), int64_t(k))) : p * kmath::ldexp(float128_t(1), int64_t(k));
}


}

#endif
