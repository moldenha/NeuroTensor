#ifndef NT_MATH_FLOAT128_TRIG_UTILS_HPP__
#define NT_MATH_FLOAT128_TRIG_UTILS_HPP__

// this is a header file to contain some trig utils
#include "../float128_impl.h"
#include "../b128.h"
#include "../b256.h"
#include "../from_string.hpp"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "../kmath/round.hpp"

// constexpr math functions
namespace nt::math::trig{

// kernels assume |x| <= pi / 4
NT_ALWAYS_INLINE constexpr float128_t sin_kernel(float128_t x) noexcept {
    float128_t x2 = x * x;

    // x - x^3/6 + x^5/120 - x^7/5040
    float128_t r = x;
    float128_t term = x * x2;

    r -= term / float128_t(6);
    term *= x2;
    r += term / float128_t(120);
    term *= x2;
    r -= term / float128_t(5040);

    return r;
}

NT_ALWAYS_INLINE constexpr float128_t cos_kernel(float128_t x) noexcept {
    float128_t x2 = x * x;

    // 1 - x^2/2 + x^4/24 - x^6/720
    float128_t r = float128_t(1);
    float128_t term = x2;

    r -= term / float128_t(2);
    term *= x2;
    r += term / float128_t(24);
    term *= x2;
    r -= term / float128_t(720);

    return r;
}


struct trig_reduction {
    int quadrant;
    float128_t r;
};

NT_ALWAYS_INLINE constexpr trig_reduction reduce_trig(float128_t x) noexcept {
    float128_t kf = x * f128_constants::INV_PI_2;
    constexpr float128_t half = 0.5_f128;
    // round-to-nearest
    float128_t kf_rounded = kmath::floor(kf + half);
    if (kf.get_bits().sign()) kf_rounded = kmath::ceil(kf - half);

    int64_t k = static_cast<int64_t>(kf_rounded);

    float128_t r = x - float128_t(k) * f128_constants::PI_2;

    return { static_cast<int>(k & 3), r };
}






}

#endif
