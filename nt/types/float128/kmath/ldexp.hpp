#ifndef NT_MATH_KMATH_FLOAT128_LDEXP_HPP__
#define NT_MATH_KMATH_FLOAT128_LDEXP_HPP__

#include "../float128_impl.h"
#include "../../../math/kmath/ldexp_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{


template<>
inline constexpr float128_t ldexp(float128_t x, int n) noexcept {
    auto b = x.get_bits();

    const bool sign = b.sign();
    const uint16_t exp  = b.exponent();
    const b128 mant = b.mantissa();
    // --- 1. Zero → return zero ---
    if (exp == 0 && mant.is_zero())
        return x;

    // --- 2. Inf or NaN → return as-is ---
    if (exp == 0x7FFF)
        return x;

    // --- 3. Normal x ---
    if (exp != 0) {
        // attempt to add n to exponent
        int new_exp = int(exp) + n;

        // Overflow → return infinity
        if (new_exp >= 0x7FFF)
            return float128_t::make_inf(sign);

        // Normal case
        if (new_exp > 0)
            return float128_t(float128_bits::pack(sign, uint16_t(new_exp), mant));

        // Now must produce a subnormal
        // new_exp <= 0
        // We need to shift the mantissa right to encode subnormals
        int shift = 1 - new_exp;   // amount to shift toward right
        if (shift >= 113) {
            // too small: underflows to zero
            return float128_t(float128_bits::pack(sign, 0, b128(0)));
        }

        b128 new_mant = (b128(1) << 112) | mant;
        new_mant >>= shift;
        return float128_t(float128_bits::pack(sign, 0, new_mant));
    }

    // --- 4. Subnormal x ---
    // We need to renormalize then continue as normal
    // Compute the effective exponent for subnormals = 1-bias
    int e = 1 - 16383;  

    b128 m = mant;
    int leading = 0;

    while ((m >> 111).is_zero()) {
        m <<= 1;
        leading++;
    }

    // value == (1.m) * 2^(e - leading)
    e -= leading;

    // Now add n
    e += n;

    int final_exp = e + 16383;

    // Overflow
    if (final_exp >= 0x7FFF)
        return float128_t::make_inf(sign);

    // Underflow to subnormal or zero
    if (final_exp <= 0) {
        // shift mant right to encode subnormal
        int shift = 1 - final_exp;
        if (shift >= 113)
            return float128_t(float128_bits::pack(sign, 0, b128(0))); // underflow

        b128 new_mant = m >> shift;
        return float128_t(float128_bits::pack(sign, 0, new_mant));
    }

    // Normalized result
    return float128_t(float128_bits::pack(sign, uint16_t(final_exp), m));

}





}

#endif
