#ifndef NT_MATH_FLOAT128_POW_HPP__
#define NT_MATH_FLOAT128_POW_HPP__

// x^y = e ^ (y * ln(x))

#include "../float128_impl.h"
#include "../b128.h"
#include "../b256.h"
#include "../from_string.hpp"
#include "../../../utils/always_inline_macro.h"
#include "constants.h"
#include "exp.hpp"
#include "log.hpp"
#include "../kmath/trunc.hpp"
#include "../kmath/is_integer.hpp"
#include "../kmath/ldexp.hpp"

// constexpr math functions
namespace nt::math{

namespace details{

NT_ALWAYS_INLINE bool is_odd_integer(const float128_t& v) noexcept {
    auto bits = v.get_bits();

    // if not finite false
    if (bits.is_nan() || bits.is_inf())
        return false;

    int32_t E = int32_t(bits.exponent()) - 16383;

    // Not an integer
    if (E < 0)
        return false;

    // all integers >= 2 are even if LSB bit is zero
    if (E > 112)
        return false; // multiples of 2

    b128 sig = bits.mantissa_with_hidden_bit();
    int lsb_pos = 112 - E;

    return sig.bit(lsb_pos);
}

}




inline float128_t f128_pow(float128_t x, float128_t y) noexcept {
    float128_bits xb = x.get_bits();
    float128_bits yb = y.get_bits();
    // y == 0 → 1 (even if x is NaN per IEEE)
    if(yb.is_zero()) return float128_t(1);
    if(xb.is_nan() || yb.is_nan()) return float128_t::make_nan();
    // x == 0
    if (xb.is_zero()) {
        if (yb.sign()) // yb negative
            return float128_t::make_inf(true); // 0^negative → +inf
        return float128_t::make_zero(xb.sign() && details::is_odd_integer(y));
    }

    if(kmath::is_integer(y)){
        int64_t n(y);
        bool neg = false;
        if (n < 0) {
            neg = true;
            n = -n;
        }
        float128_t result(1);
        float128_t base = x;
        while (n != 0) {
            if (n & 1)
                result *= base;
            base *= base;
            n >>= 1;
        }
        return neg ? float128_t(1) / result : result;
    }
    // x < 0
    if (xb.sign()) {
        // y must be integer
        if (!kmath::is_integer(y))
            return float128_t::make_nan();

        int64_t yt(y);
        // integer exponent
        b128 n(yt);
        bool odd = !(n & b128(1)).is_zero();

        float128_t res = ::nt::math::exp(y * ::nt::math::log(-x));
        return odd ? -res : res;
    }
    // x > 0 (normal case)
    return ::nt::math::exp(y * ::nt::math::log(x));
}


inline float128_t f128_ipow(float128_t x, int64_t y) noexcept {
    float128_bits xb = x.get_bits();
    // y == 0 → 1 (even if x is NaN per IEEE)
    if(y == 0) return float128_t(1);
    if(xb.is_nan()) return float128_t::make_nan();
    // x == 0
    if (xb.is_zero()) {
        if (y < 0)
            return float128_t::make_inf(true); // 0^negative → +inf
        // if (y & 1) -> true -> y is odd
        return float128_t::make_zero(xb.sign() && (y & 1));
    }

    bool neg = false;
    if (y < 0) {
        neg = true;
        y = -y;
    }
    float128_t result(1);
    float128_t base = x;
    while (y != 0) {
        if (y & 1)
            result *= base;
        base *= base;
        y >>= 1;
    }
    return neg ? float128_t(1) / result : result;
}


}

#endif
