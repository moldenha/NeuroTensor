#ifndef NT_MATH_KMATH_FLOAT128_IS_INTEGER_HPP__
#define NT_MATH_KMATH_FLOAT128_IS_INTEGER_HPP__

#include "../float128_impl.h"
#include "../b128.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{

inline constexpr bool is_integer(float128_t arg) noexcept {
    const float128_bits bits = arg.get_bits();

    bool sign = bits.sign();
    uint16_t exp  = bits.exponent();
    b128 mant = bits.mantissa();

    // Check special cases
    if (exp == 0x7FFF)  // inf or NaN
        return false;

    int32_t E = int32_t(exp) - 16383;   // unbiased exponent
    // Case 1: |x| < 1
    if (E < 0) {
        return false;
    }
    
    // Case 2: integer already (no fractional bits)
    if (E >= 112) {
        return true;
    }

    return false;
}



}

#endif
