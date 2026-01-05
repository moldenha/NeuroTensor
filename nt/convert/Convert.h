#ifndef NT_CONVERT_SUPPORTED_TYPES_H__
#define NT_CONVERT_SUPPORTED_TYPES_H__

#include "../Tensor.h"
#include "../utils/api_macro.h"
#include "../utils/type_traits.h"
#include "../dtype/compatible/DType_compatible.h"
#include "../dtype/compatible/DTypeDeclareMacros.h"
#include "../types/Types.h"
#include "../bit/bit_cast.h"
#include "utils.h"
#include "floating.h"
#include <type_traits>
#include <cmath>

namespace nt::convert{

namespace details{

/* template<typename T> */
/* inline static constexpr bool valid_convert_type_v = */
/*     (type_traits::is_same_v<nt::type_traits::remove_cvref_t<T>, uint_bool_t> || */
/*      type_traits::is_same_v<nt::type_traits::remove_cvref_t<T>, bool> || */
/*      (DTypeFuncs::type_to_dtype<nt::type_traits::remove_cvref_t<T>> != DType::Bool)); */

NT_ALWAYS_INLINE int128_t float32_to_int128(float value) {
    // Handle special cases
    if (std::isnan(value)) {
        throw std::invalid_argument("Cannot convert NaN to int128");
    }
    if (std::isinf(value)) {
        throw std::overflow_error("Cannot convert infinity to int128");
    }
    if (value > static_cast<float>(std::numeric_limits<int128_t>::max())) {
        throw std::overflow_error("Float value is too large for int128");
    }
    if (value < static_cast<float>(std::numeric_limits<int128_t>::min())) {
        throw std::overflow_error("Float value is too small for int128");
    }

    // Convert the float to int128
    return static_cast<int128_t>(value);
}

NT_ALWAYS_INLINE uint128_t float32_to_uint128(float value) {
    // Handle special cases
    if (std::isnan(value)) {
        throw std::invalid_argument("Cannot convert NaN to uint128");
    }
    if (std::isinf(value)) {
        throw std::overflow_error("Cannot convert infinity to uint128");
    }
    if (value > static_cast<float>(std::numeric_limits<uint128_t>::max())) {
        throw std::overflow_error("Float value is too large for uint128");
    }
    if (value < static_cast<float>(std::numeric_limits<uint128_t>::min())) {
        throw std::overflow_error("Float value is too small for uint128");
    }

    // Convert the float to int128
    return static_cast<uint128_t>(value);
}


// template<typename U>
// NT_ALWAYS_INLINE float16_t float32_to_float16(U&& _val){
//     uint32_t f32u = bit_cast<uint32_t>(std::forward<U>(_val));
//     static constexpr uint32_t f32u_infty = uint32_t(255) << 23;
//     static constexpr uint32_t f16u_max = (uint32_t(127) + uint32_t(16)) << 23;
//     static constexpr uint32_t denorm_magic =
//       ((uint32_t(127) - uint32_t(15)) + (uint32_t(23) - uint32_t(10)) + uint32_t(1)) << 23;
//     uint16_t f16u;

//     uint32_t sign = f32u & (uint32_t(1) << 31);
//     f32u ^= sign;

//     if (f32u > FLOAT16_MAX) { f16u = (f32u > f32u_infty) ?  0x7e00 : 0x7c00; } // inf
//     else { 
//       if (f32u < (uint32_t(113) << 23)) { 
//         f32u = bit_cast<uint32_t>(bit_cast<float>(f32u) + bit_cast<float>(denorm_magic));

//         f16u = static_cast<uint16_t>(f32u - denorm_magic);
//       } else {
//         uint32_t mant_odd = (f32u >> 13) & 1;

//         f32u += (static_cast<uint32_t>(15 - 127) << 23) + uint32_t(0xfff);
//         f32u += mant_odd;
//         f16u = static_cast<uint16_t>(f32u >> 13);
//       }
//     }
//     f16u |= sign >> 16;
//     return bit_cast<float16_t>(f16u);

// }

// template<typename U>
// NT_ALWAYS_INLINE float float16_to_float32(U&& _val){
//     uint16_t f16u = bit_cast<uint16_t>(std::forward<U>(_val));
//     const float denorm_magic = bit_cast<float>((uint32_t(113) << 23));
//     static constexpr uint32_t shifted_exp = uint32_t(0x7c00) << 13;
//     uint32_t f32u;
//     f32u = (f16u & uint32_t(0x7fff)) << 13; //exponent / mantista
//     uint32_t exp = shifted_exp & f32u; //just exponent
//     f32u += (uint32_t(127) - uint32_t(15)) << 23; // exponent adjust
//     if (exp == shifted_exp) // Inf or NaN
//         f32u += (uint32_t(128) - uint32_t(16)) << 23;
//     else if (exp == 0) {
//         f32u += (1) << 23;
//         f32u = bit_cast<uint32_t>(bit_cast<float>(f32u) - denorm_magic);
//     }
//     f32u |= (f16u & uint32_t(0x8000)) << 16; // sign
//     return bit_cast<float>(f32u);
// }

// Below is converting from a 128 bit integer to floating points of precision between 16 and 128
// There is going to be a loss of information for pretty much every single floating type if 128 bit integer was needed for processing
// float128_t is obviously the closest you will get to keeping all of the information

template<typename T>
NT_ALWAYS_INLINE T portable_128_int_to_floating(int128_t val){
    return static_cast<T>(val);
}

template<typename T>
NT_ALWAYS_INLINE T portable_128_int_to_floating(uint128_t val){
    return static_cast<T>(val);
}


template<typename T>
inline static constexpr T TWO_POW_64 = static_cast<T>(1ULL << 63) * T(2);


template<>
NT_ALWAYS_INLINE float portable_128_int_to_floating<float>(int128_t val) {
    // Convert manually via double, because int128 -> float causes ___floattihf
    constexpr int64_t high_mask = ~0ULL;
    int64_t high = static_cast<int64_t>(val >> 64);
    uint64_t low = static_cast<uint64_t>(val);

    double result = static_cast<double>(high) * TWO_POW_64<double> +
                    static_cast<double>(low);
    return static_cast<float>(result);
}

template<>
NT_ALWAYS_INLINE double portable_128_int_to_floating<double>(int128_t val) {
    int64_t high = static_cast<int64_t>(val >> 64);
    uint64_t low = static_cast<uint64_t>(val);

    double result = static_cast<double>(high) * TWO_POW_64<double> +
                    static_cast<double>(low);
    return result;
}

template<>
inline nt::float128_t portable_128_int_to_floating(int128_t val) {
    bool sign = val < 0;
    val = sign ? -val : val;
    nt::b128 b = nt::bit_cast<nt::b128>(val);
    return sign ? -nt::float128_t::from_integer(b) : nt::float128_t::from_integer(b);
}




template<>
NT_ALWAYS_INLINE float portable_128_int_to_floating<float>(uint128_t val) {
    uint64_t high = static_cast<uint64_t>(val >> 64);
    uint64_t low = static_cast<uint64_t>(val);

    double result = static_cast<double>(high) * TWO_POW_64<double> +
                    static_cast<double>(low);
    return static_cast<float>(result);
}

template<>
NT_ALWAYS_INLINE double portable_128_int_to_floating<double>(uint128_t val) {

    uint64_t high = static_cast<uint64_t>(val >> 64);
    uint64_t low = static_cast<uint64_t>(val);

    double result = static_cast<double>(high) * TWO_POW_64<double> +
                    static_cast<double>(low);
    return result;
}

template<>
inline float128_t portable_128_int_to_floating(uint128_t val) {
    b128 b = bit_cast<b128>(val);
    return float128_t::from_integer(b);
}



template<>
NT_ALWAYS_INLINE float16_t portable_128_int_to_floating<float16_t>(int128_t val) {
    // Convert manually via double, because int128 -> float causes ___floattihf
    constexpr int64_t high_mask = ~0ULL;
    int64_t high = static_cast<int64_t>(val >> 64);
    uint64_t low = static_cast<uint64_t>(val);

    double result = static_cast<double>(high) * TWO_POW_64<double> +
                    static_cast<double>(low);
    return safe_float16_from_double(result);
}


template<>
NT_ALWAYS_INLINE float16_t portable_128_int_to_floating<float16_t>(uint128_t val) {
    uint64_t high = static_cast<uint64_t>(val >> 64);
    uint64_t low = static_cast<uint64_t>(val);

    double result = static_cast<double>(high) * TWO_POW_64<double> +
                    static_cast<double>(low);
    return safe_float16_from_double(result);
}



}

template<typename To, typename _From,
    std::enable_if_t<details::valid_convert_type_v<To>
                    && details::valid_convert_type_v<_From>, bool> = true>
NT_ALWAYS_INLINE nt::type_traits::decay_t<To> convert(_From&& f){
    using From = typename nt::type_traits::decay_t<_From>;
    static constexpr DType FromDType = DTypeFuncs::type_to_dtype<From>;
    static constexpr DType ToDType = DTypeFuncs::type_to_dtype<To>;
    if constexpr (FromDType == ToDType){
        return std::forward<_From>(f);
    }
    else if constexpr (type_traits::is_same_v<From, Tensor>){
        return std::forward<_From>(f).toScalar().template to<To>();
    }
    else if constexpr (type_traits::is_same_v<To, Tensor>){
        return Tensor(Scalar(std::forward<_From>(f)));
    }
    //handling all complex cases
    else if constexpr (::nt::type_traits::is_complex<From>::value && ::nt::type_traits::is_complex<To>::value){
        if constexpr (type_traits::is_same_v<typename From::value_type, double> && type_traits::is_same_v<typename To::value_type, float16_t>){
            To(details::safe_float16_from_double(std::get<0>(std::forward<_From>(f))), details::safe_float16_from_double(std::get<1>(std::forward<_From>(f))));
        }
        return To(std::forward<_From>(f));
    }
    else if constexpr (::nt::type_traits::is_complex<From>::value){
        return convert<To>(std::get<0>(std::forward<_From>(f)));
    }
    else if constexpr (::nt::type_traits::is_complex<To>::value){
        return To(convert<typename To::value_type>(std::forward<_From>(f)), typename To::value_type(0));
    }
    //handling bool cases:
    else if constexpr (type_traits::is_same_v<From, bool>){
        return convert<To>(std::forward<_From>(f) ? float(1) : float(0));
    }
    else if constexpr (type_traits::is_same_v<From, uint_bool_t>){
        return convert<To>((std::forward<_From>(f).value == 1) ? float(1) : float(0));
    }
    else if constexpr (type_traits::is_same_v<To, bool>){
        From zero(0);
        return zero < std::forward<_From>(f);
    }
    else if constexpr (type_traits::is_same_v<To, uint_bool_t>){
        From zero(0);
        return uint_bool_t(zero < std::forward<_From>(f));
    }

    //some specific use cases pre-defined
    else if constexpr (type_traits::is_same_v<From, float> && type_traits::is_same_v<To, int128_t>){
        return details::float32_to_int128(std::forward<_From>(f)); 
    }
    else if constexpr (type_traits::is_same_v<From, float> && type_traits::is_same_v<To, uint128_t>){
        return details::float32_to_uint128(std::forward<_From>(f)); 
    }
    else if constexpr(FromDType == DType::Float16){
        float val = _NT_FLOAT16_TO_FLOAT32_(std::forward<_From>(f));
        return convert::convert<To>(val);
    }
    else if constexpr(ToDType == DType::Float16){
        if constexpr (FromDType == DType::Float64){
            return details::safe_float16_from_double(std::forward<_From>(f));
        }
        float sub_val = convert::convert<float>(std::forward<_From>(f));
        return _NT_FLOAT32_TO_FLOAT16_(sub_val);
    }
    else if constexpr (DTypeFuncs::is_dtype_floating_v<FromDType> && type_traits::is_same_v<To, int128_t>){
        return details::float32_to_int128(convert<float>(std::forward<_From>(f)));
    }
    else if constexpr (std::is_floating_point_v<From> && type_traits::is_same_v<To, uint128_t>){
        return details::float32_to_uint128(convert<float>(std::forward<_From>(f)));
    }
    else if constexpr (type_traits::is_same_v<To, float128_t>){
        if constexpr (type_traits::is_decay_in_v<From, uint128_t, int128_t>){
            return details::portable_128_int_to_floating<To>(std::forward<_From>(f));
        }else if constexpr (type_traits::is_integral_v<From>){
            int64_t b = convert<int64_t>(std::forward<_From>(f));
            int64_t ab = b < 0 ? -b : b;
            return b < 0 ? -float128_t::from_integer(b128(ab)) : float128_t::from_integer(b128(ab));
        }
        else{
            return float128_t(convert<double>(std::forward<_From>(f)));
        }
    }
    else if constexpr (type_traits::is_same_v<From, float128_t>){
        if constexpr (type_traits::is_integral_v<To>){
            return convert<To>(int64_t(f));
        }else{
            return convert<To>(double(f));
        }
    }
#ifndef __SIZEOF_INT128__
    else if constexpr (type_traits::is_same_v<From, int128_t>){
        return convert<To>(int64_t(std::forward<_From>(f)));
    }
    else if constexpr (type_traits::is_same_v<From, uint128_t>){
        return convert<To>(uint64_t(std::forward<_From>(f)));
    }
    else if constexpr (type_traits::is_same_v<To, int128_t>){
        return int128_t(convert<int64_t>(std::forward<_From>(f)));
    }
    else if constexpr (type_traits::is_same_v<To, uint128_t>){
        return uint128_t(convert<uint64_t>(std::forward<_From>(f)));
    }
#else
    else if constexpr(type_traits::is_same_v<From, int128_t>){
        return convert<To>(int64_t(std::forward<_From>(f)));
    }
    else if constexpr(type_traits::is_same_v<From, uint128_t> && DTypeFuncs::is_dtype_floating_v<ToDType> ){
        return details::portable_128_int_to_floating<To>(std::forward<_From>(f)); 
    }
#endif
    else{
        return static_cast<To>(std::forward<_From>(f));
    }
}

template<DType dt, typename A, std::enable_if_t<details::valid_convert_type_v<A>, bool> = true>
NT_ALWAYS_INLINE DTypeFuncs::dtype_to_type_t<dt> convert(A&& val){
    return convert<DTypeFuncs::dtype_to_type_t<dt>>(std::forward<A>(val)); 
}

namespace test{

NEUROTENSOR_API void test_convert();
}

}

#endif // NT_CONVERT_DTYPE_H__
