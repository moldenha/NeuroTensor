#ifndef NT_MATH_FUNCTIONAL_ROUND_HPP__
#define NT_MATH_FUNCTIONAL_ROUND_HPP__

#include "utils.h"
#include "general_include.h"
#include "trunc.hpp"


#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(round) // <nt/types/float128/math/round.hpp>


namespace nt::math{

namespace round_details{
template <class Float>
inline Float round_custom(const Float x) noexcept {
    using FB   = nt::float_bits<Float>;
    using UInt = typename FB::integer_type;

    constexpr std::size_t E    = FB::exponent_bits;
    constexpr std::size_t M    = FB::mantissa_bits;
    constexpr std::size_t Bias = FB::exponent_bias;

    UInt bits = nt::bit_cast<UInt>(x);

    // Extract exponent
    int32_t exponent = int((bits >> M) & ((UInt(1) << E) - 1));
    UInt mantissa = bits & ((UInt(1) << M) - 1);

    // Handle NaN/inf/zero (fast)
    constexpr UInt exp_all_ones = ((UInt(1) << E) - 1);
    if (exponent == exp_all_ones) return x;
    if ((bits & ~((UInt(1) << (E + M)))) == 0) return x;

    // Unbias exponent
    int32_t e = exponent - Bias;

    // |x| < 1 → round to ±1 or ±0 depending on |x| >= 0.5
    if (e < 0) {
        Float absx = x < Float(0) ? -x : x;
        if (absx < Float(0.5)) {
            return Float(0) * (x < 0 ? Float(-1) : Float(1));
        } else {
            return (x < 0) ? Float(-1) : Float(1);
        }
    }

    // If exponent is large enough, already integer
    if (e >= M)
        return x;

    // Compute rounding mask
    int frac_bits = M - e;
    UInt frac_mask = (UInt(1) << frac_bits) - 1;
    UInt half_bit  = UInt(1) << (frac_bits - 1);

    UInt rounded = bits;

    // Add the half ULP
    rounded += half_bit;

    // Clear fractional bits
    rounded &= ~frac_mask;

    return nt::bit_cast<Float>(rounded);
}


}

inline float16_t round(const float16_t& val) noexcept {
    return round_details::round_custom(val);
}

NT_ALWAYS_INLINE float round(const float& val){
    return std::roundf(val);
}

NT_ALWAYS_INLINE double round(const double& val){
    return std::round(val);
}

#define NT_ROUND_INTEGER_(type, name_a, name_b)\
NT_ALWAYS_INLINE type round(const type& val) noexcept { return val; }\


#define NT_ROUND_COMPLEX_(type, name_a, name_b)\
NT_ALWAYS_INLINE type round(const type& val) noexcept {\
    return type(round(val.real()), round(val.imag())); \
}\

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_ROUND_INTEGER_) 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_ROUND_INTEGER_) 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_ROUND_COMPLEX_) 

#undef NT_ROUND_INTEGER_ 
#undef NT_ROUND_COMPLEX_ 

}

#endif
