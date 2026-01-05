/*
 * This is a specific header file for making floating types convertible between each other
 * This is partially built so that the types/complex.h can include this file without worrying about making dependency errors
*/

#ifndef NT_CONVERT_FLOATING_CONVERT_H__
#define NT_CONVERT_FLOATING_CONVERT_H__


#include "../utils/type_traits.h"
#include "../types/float16.h"
#include "../types/float128.h"
#include "../utils/always_inline_macro.h"


namespace nt::convert::details{
// Constants for float16 limits
constexpr double FLOAT16_MAX = 65504.0;              // max normal
constexpr double FLOAT16_MIN_POSITIVE = 6.10e-5;     // min positive normal
constexpr double FLOAT16_SMALLEST = 5.96e-8;         // min subnormal
constexpr double FLOAT16_INFINITY = ::nt::type_traits::numeric_limits<float>::infinity();



// TODO: Make this constexpr in the future (maybe, no immediate reason right now)
NT_ALWAYS_INLINE float16_t safe_float16_from_double(const double& x) {
    if (std::isnan(x)) return 0x7e00; // float16 canonical quiet NaN
    if (std::isinf(x)) return (x > 0) ? 0x7c00 : 0xfc00;
    if (x > FLOAT16_MAX) return 0x7c00; // +inf
    if (x < -FLOAT16_MAX) return 0xfc00; // -inf
    if (std::abs(x) < FLOAT16_SMALLEST) return 0x0000; // flush to zero

    // Convert through float32 first, which is common in practice
    float f32 = static_cast<float>(x);

    // Now convert to float16 bits
    uint32_t bits;
    std::memcpy(&bits, &f32, sizeof(bits));

    uint16_t sign     = (bits >> 16) & 0x8000;
    int16_t  exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x007FFFFF;

    if (exponent <= 0) {
        // Subnormal or underflow
        return sign;
    } else if (exponent >= 31) {
        // Overflow to infinity
        return sign | 0x7C00;
    }
    uint16_t float16_raw = sign | (exponent << 10) | (mantissa >> 13);
    float16_t out;
    std::memcpy(&out, &float16_raw, sizeof(out));
    return out;
}




}


namespace nt::convert{

template<typename Out, typename In, 
        std::enable_if_t<type_traits::is_floating_point_v<In> && type_traits::is_floating_point_v<Out>,
                            bool> = true>
NT_ALWAYS_INLINE Out floating_convert(const In& val){
    if constexpr (type_traits::is_same_v<In, Out>){
        return val;
    }
    else if constexpr (type_traits::is_same_v<float16_t, Out> && type_traits::is_same_v<double, In>){
        return details::safe_float16_from_double(val);
    }
    else if constexpr (type_traits::is_same_v<float16_t, Out>){
        return _NT_FLOAT32_TO_FLOAT16_(floating_convert<float>(val));
    }
    else if constexpr (type_traits::is_same_v<float16_t, In>){
        return floating_convert<Out>(_NT_FLOAT16_TO_FLOAT32_(val));
    }
    else if constexpr (type_traits::is_same_v<float128_t, In>){
        if constexpr (type_traits::is_same_v<double, Out>){
            return double(val);
        }else if constexpr (type_traits::is_same_v<float, Out>){
            return float(val);
        }else{
            return floating_convert<Out>(double(val));
        }
    }
    else if constexpr (type_traits::is_same_v<float128_t, Out>){
        return float128_t(floating_convert<double>(val));
    }
    else{
        return static_cast<Out>(val);
    }
}

}

#endif
