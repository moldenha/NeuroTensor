#ifndef NT_MATH_FUNCTIONAL_NAN_HPP__
#define NT_MATH_FUNCTIONAL_NAN_HPP__

// nan
// isnan
//
// The reason isnan is not constexpr for floats:
//  - the kmath::isnan is constexpr
//  - However, this is slower on non-constexpr input variables, than the below isnan is
//  - Therefore, it is up to the user to decide if a constexpr isnan is needed and use the kmath::isnan
//  The below isnan can sometimes peform better than std::isnan
//      it is branchless, and generalizes to work for any floating type that is supported by ::nt::float_bits
//          - to make a float work with ::nt::float_bits just define 
//              ::nt::type_traits::numeric_num_digits<floating_type> as either std::numerical_limits::digits or the number of mantissa bits
//              in the float plus 1

#include "../../bit/bitset.h"
#include "../../bit/float_bits.h"
#include "../../bit/kbit_cast.h"
#include "../../bit/bit_cast.h"
#include "../kmath/nan.hpp"
#include "../../utils/always_inline_macro.h"
#include "../../dtype/compatible/DTypeDeclareMacros.h"
#include "nan_decl.h"
#include "utils.h"
#include <cstdint>


namespace nt::math{


template<typename T>
constexpr T nan() noexcept;



template<>
inline constexpr float16_t nan<float16_t>() noexcept {
    return kmath::generate_qNaN<float16_t>();
}

template<>
inline constexpr float nan<float>() noexcept {
    return kmath::generate_qNaN<float>();
}

template<>
inline constexpr double nan<double>() noexcept {
    return kmath::generate_qNaN<double>();
}

inline bool isnan(const float16_t& v) noexcept {
    using UInt = typename float_bits<float16_t>::integer_type;
    constexpr int ebits = float_bits<float16_t>::exponent_bits;
    constexpr int mbits = float_bits<float16_t>::mantissa_bits;
    UInt bits = bit_cast<UInt>(v);
    constexpr UInt sign_mask = (UInt(1) << (ebits + mbits)) - (UInt(1) << (ebits + mbits));
    constexpr UInt exponent_mask = ((UInt(1) << ebits) - 1) << mbits;
    UInt exponent = bits & exponent_mask;
    constexpr UInt mantissa_mask = (UInt(1) << mbits) - 1;
    UInt mantissa = bits & mantissa_mask;
    return (exponent == exponent_mask) && (mantissa != 0);
}


inline bool isnan(const float& v) noexcept {
    using UInt = typename float_bits<float>::integer_type;
    constexpr int ebits = float_bits<float>::exponent_bits;
    constexpr int mbits = float_bits<float>::mantissa_bits;
    UInt bits = bit_cast<UInt>(v);
    constexpr UInt sign_mask = (UInt(1) << (ebits + mbits)) - (UInt(1) << (ebits + mbits));
    constexpr UInt exponent_mask = ((UInt(1) << ebits) - 1) << mbits;
    UInt exponent = bits & exponent_mask;
    constexpr UInt mantissa_mask = (UInt(1) << mbits) - 1;
    UInt mantissa = bits & mantissa_mask;
    return (exponent == exponent_mask) && (mantissa != 0);
}

inline bool isnan(const double& v) noexcept {
    using UInt = typename float_bits<double>::integer_type;
    constexpr int ebits = float_bits<double>::exponent_bits;
    constexpr int mbits = float_bits<double>::mantissa_bits;
    UInt bits = bit_cast<UInt>(v);
    constexpr UInt sign_mask = (UInt(1) << (ebits + mbits)) - (UInt(1) << (ebits + mbits));
    constexpr UInt exponent_mask = ((UInt(1) << ebits) - 1) << mbits;
    UInt exponent = bits & exponent_mask;
    constexpr UInt mantissa_mask = (UInt(1) << mbits) - 1;
    UInt mantissa = bits & mantissa_mask;
    return (exponent == exponent_mask) && (mantissa != 0);
}




#define NT_NAN_INTEGER_(type, name_a, name_b)\
template<>\
inline constexpr type nan() noexcept {\
    return 0;\
}\
inline constexpr bool isnan(const type& val) noexcept {\
    return false;\
}

#define NT_NAN_COMPLEX_(type, name_a, name_b)\
template<>\
inline constexpr type nan() noexcept {\
    constexpr type::value_type val = nan<type::value_type>();\
    return type(val, val);\
}\
inline bool isnan(const type& val) noexcept {\
    return isnan(val.real()) && isnan(val.imag());\
}

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_NAN_INTEGER_);
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_NAN_INTEGER_); 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_NAN_COMPLEX_);

#undef NT_NAN_INTEGER_ 
#undef NT_NAN_COMPLEX_ 


}


#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(nan) // <nt/types/float128/math/nan.hpp>

#endif
