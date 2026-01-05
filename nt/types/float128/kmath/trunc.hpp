#ifndef NT_MATH_KMATH_FLOAT128_TRUNC_HPP__
#define NT_MATH_KMATH_FLOAT128_TRUNC_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../../../math/kmath/trunc_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{

template<>
inline constexpr float128_t trunc(float128_t arg) noexcept {
    const float128_bits bits = arg.get_bits();

    bool sign = bits.sign();
    uint16_t exp  = bits.exponent();
    b128 mant = bits.mantissa();

    // Check special cases
    if (exp == 0x7FFF)  // inf or NaN
        return arg;       // per C spec, trunc(NaN)=NaN, trunc(±inf)=±inf

    int32_t E = int32_t(exp) - 16383;   // unbiased exponent
    // Case 1: |x| < 1
    if (E < 0) {
        // return ±0
        return float128_t::make_zero(sign);
    }
    
    // Case 2: integer already (no fractional bits)
    if (E >= 112) {
        return arg;   // nothing to do
    }

    // Case 3: fractional part present
    // integer bits = 1 (implicit) + top E bits of mantissa
    // fractional bits = bottom 112 - E bits

    int frac_bits = 112 - E;

    // Create mask: keep high (112 - frac_bits) = E bits
    b128 mask = (~b128(0)) << frac_bits;

    // Apply mask to mantissa
    b128 mant_intpart = mant & mask;

    return float128_t(float128_bits::pack(sign, exp, mant_intpart));
}



}

#endif
