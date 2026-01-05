#ifndef NT_MATH_FUNCTIONAL_TRUNC_HPP__
#define NT_MATH_FUNCTIONAL_TRUNC_HPP__

// trunc

#include "utils.h"
#include "general_include.h"
#include "../../bit/bitset.h"
#include "../../bit/float_bits.h"
#include "../../bit/kbit_cast.h"
#include "../../bit/bit_cast.h"

#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(trunc) // <nt/types/float128/math/trunc.hpp>

namespace nt::math{

namespace trunc_details{

template <class Float>
inline Float trunc_custom_branchless(Float x) noexcept {
    using FB   = ::nt::float_bits<Float>;
    using UInt = typename FB::integer_type;

    constexpr std::size_t E    = FB::exponent_bits;
    constexpr std::size_t M    = FB::mantissa_bits;
    constexpr std::size_t Bias = FB::exponent_bias;

    UInt bits = ::nt::bit_cast<UInt>(x);

    // Extract exponent and mantissa
    UInt exponent = (bits >> M) & ((UInt(1) << E) - 1);
    UInt mantissa = bits & ((UInt(1) << M) - 1);

    // Check classifications (branchless masks)
    constexpr UInt exp_all_ones  = ((UInt(1) << E) - 1);
    UInt is_inf_or_nan = UInt(exponent == exp_all_ones);
    UInt is_zero       = UInt((bits & ~((UInt(1) << (E + M)))) == 0);

    // Effective unbiased exponent
    // (signed range fits into int32)
    int e = int(exponent) - Bias;

    // e < 0  → |x| < 1 → ±0
    UInt mask_lt1 = UInt(e < 0);

    // e >= M → already integer
    UInt mask_int = UInt(e >= M);

    // Number of fractional bits to clear (valid only if 0 <= e < M)
    int32_t frac_bits = M - e;
    UInt frac_mask = (UInt(1) << frac_bits) - 1;

    // Mask to zero out fractional part
    UInt new_bits = bits & ~frac_mask;

    // Select in branchless fashion:
    // pref_int_or_nan_or_zero ? bits : truncated_bits
    UInt select_trunc = ~( (mask_lt1 | mask_int | is_inf_or_nan | is_zero) - 1 );

    // If |x| < 1 → return ±0
    UInt sign_bit = bits & (UInt(1) << (E + M));
    UInt zero_bits = sign_bit;  // preserves sign of zero

    // Build the result:
    //     if |x|<1 → signed zero
    // else if integer/inf/nan/zero → original bits
    // else → truncated bits
    UInt result_bits =
        (mask_lt1 ? zero_bits : 0) |
        (!mask_lt1 ? ((select_trunc ? bits : new_bits)) : 0);

    return ::nt::bit_cast<Float>(result_bits);
}


}

inline float16_t trunc(const float16_t& val) noexcept {
    return trunc_details::trunc_custom_branchless<float16_t>(val);
}

NT_ALWAYS_INLINE float trunc(const float& val) noexcept {
    return std::truncf(val);
}

NT_ALWAYS_INLINE double trunc(const double& val) noexcept {
    return std::trunc(val);
}

#define NT_TRUNC_INTEGER_(type, name_a, name_b)\
NT_ALWAYS_INLINE constexpr type trunc(const type& val) noexcept {\
    return val;\
}\

#define NT_TRUNC_COMPLEX_(type, name_a, name_b)\
NT_ALWAYS_INLINE type trunc(const type& val) noexcept {\
    return type(trunc(val.real()), trunc(val.imag()));\
}\


NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_TRUNC_INTEGER_) 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_TRUNC_INTEGER_) 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_TRUNC_COMPLEX_) 

#undef NT_TRUNC_INTEGER_ 
#undef NT_TRUNC_COMPLEX_ 


}

#endif
