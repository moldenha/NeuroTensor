#ifndef NT_MATH_KMATH_FLOAT128_FREXP_HPP__
#define NT_MATH_KMATH_FLOAT128_FREXP_HPP__

#include "../float128_impl.h"
#include "../../../math/kmath/frexp_decl.h"
#include "../../../utils/type_traits.h"
#include "../../../utils/always_inline_macro.h"

// constexpr math functions
namespace nt::math::kmath{


template<>
inline constexpr float128_t frexp(float128_t x, int* o) {
    auto b = x.get_bits();

    const bool sign = b.sign();
    const uint16_t exp  = b.exponent();
    const b128 mant = b.mantissa();

    // 1. Zero → return (0, 0)
    if (exp == 0 && mant == 0){
        if(o != nullptr) *o = 0;
        return x;
    }

    // 2. Inf or NaN → frexp returns x, exponent = 0
    if (exp == 0x7FFF){
        if(o != nullptr) *o = 0;
        return x;
    }

    // 3. Normal numbers
    if (exp != 0) {
        int e = int(exp) - 16383;            // unbiased exponent
        // Set exponent to bias-1 so m ∈ [0.5,1)
        uint16_t new_exp = 16383 - 1;
        if(o != nullptr) *o = e + 1;
        return float128_t(float128_bits::pack(sign, new_exp, mant));
    }

    // 4. Subnormal numbers: need to normalize
    // Shift mantissa left until hidden leading 1 appears at bit 112
    b128 m = mant;
    int shift = 0;
    while ((m >> 111).is_zero()) {   // bit 112 is MSB of mantissa
        m <<= 1;
        shift++;
    }

    // after normalization, exponent = 1 - bias
    int e = 1 - 16383 - shift;

    // Now set exponent so mantissa ∈ [0.5, 1.0)
    uint16_t new_exp = 16383 - 1;
    if(o != nullptr) *o = e + 1;
    
    return float128_t(float128_bits::pack(sign, new_exp, m));
}





}

#endif
