// this is a header file for general functions to be implemented
#include <cmath>
#include "bit_128_integer.h"
#include "float16.h"
#include "float128.h"
#include "../dtype/compatible/DTypeDeclareMacros.h"
#include "../convert/Convert.h"
#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"
#include "../dtype/compatible/DType_compatible.h"

/*
This header file is meant to act as a standard math functions for 
 all supported types to run on the cpu for scalars


Even for user use cases, if a user wants to use nt::float16_t for example, there is gurenteed
    Support for the math functions below:

nt::sqrt(T)
nt::pow(T, (T or Integer)

*/


namespace nt{
namespace scalar_math_details{

template<DType dt>
struct integer_in_floating_out{
    using t = typename std::conditional<
        DTypeFuncs::is_convertible_to_floating<dt>,
        DTypeFuncs::dtype_to_type_t<DTypeFuncs::convert_to_floating<dt>>,
        float
    >::type;
};

#define X(type, dtna, dtnb) \
NT_ALWAYS_INLINE typename integer_in_floating_out<DType::dtna>::t \
integer_to_floating(const type& i) { \
    using out_type = typename integer_in_floating_out<DType::dtna>::t; \
    return convert::convert<out_type>(i); \
}

NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_  
#undef X

}

#define NT_FUNCTIONAL_CONVERT_FUNC_CONVERT(my_type, other_type, function, name)\
    convert::convert<my_type>(function(convert::convert<other_type>(name)))


// this is a macro meant to be used after all the floating type definitions
// have been called
// look at the sqrt use for an example
#define NT_FUNCTIONAL_MASS_INTEGER_CONVERT(type, a, b, func_name)\
NT_ALWAYS_INLINE type func_name(const type& i){\
    return convert::convert<type>(func_name(scalar_math_details::integer_to_floating(i)));\
}

// SQRT
NT_ALWAYS_INLINE ::nt::float16_t sqrt(const ::nt::float16_t& f){
    return NT_FUNCTIONAL_CONVERT_FUNC_CONVERT(::nt::float16_t, float, std::sqrt, f);
    return convert::convert<::nt::float16_t>(
            std::sqrt(convert::convert<float>(f)));
}

NT_ALWAYS_INLINE float sqrt(const float& f){
    return std::sqrt(f);
}

NT_ALWAYS_INLINE double sqrt(const double& f){
    return std::sqrt(f);
}

NT_ALWAYS_INLINE ::nt::float128_t sqrt(const ::nt::float128_t& f){
    return NT_FUNCTIONAL_CONVERT_FUNC_CONVERT(::nt::float128_t, double, std::sqrt, f);
}

#define X(type, a, b)\
NT_ALWAYS_INLINE type sqrt(const type& f){\
    return type(sqrt(f.real()), sqrt(f.imag()));\
}
NT_GET_X_COMPLEX_DTYPES_
#undef X

// this is a macro to make the sqrt function for all integer types
#define NT_CUR_FUNC__(type, a, b)\
    NT_FUNCTIONAL_MASS_INTEGER_CONVERT(type, a, b, sqrt)

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CUR_FUNC__);
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CUR_FUNC__);

#undef NT_CUR_FUNC__

//SQRT END

namespace details{
template<typename O, typename T>
inline constexpr bool is_powable_num = type_traits::is_same_v<O, T> || type_traits::is_integral_v<T>;
}


template<typename U, std::enable_if_t<details::is_powable_num<float, U>, bool> = true>
NT_ALWAYS_INLINE float pow(const float& a, const U& b){
    return std::pow(a, b);
}

template<typename U, std::enable_if_t<details::is_powable_num<double, U>, bool> = true>
NT_ALWAYS_INLINE double pow(const double& a, const U& b){
    return std::pow(a, b);
}


// this is so that if boost is not defined, there is not an error
// from the compiler when evaluating boost::multiprecision::pow
#ifndef BOOST_MP_STANDALONE
namespace boost::multiprecision{
template<typename T, typename U>
NT_ALWAYS_INLINE T pow(const T& t, const U& u){return ::std::pow(t, u);}
}
#endif


template<typename U, std::enable_if_t<details::is_powable_num<::nt::float128_t, U>, bool> = true>
NT_ALWAYS_INLINE ::nt::float128_t pow(const ::nt::float128_t& a, const U& b){
    if constexpr (type_traits::system_float128){
        return std::pow(a, b);
    }else{
        return boost::multiprecision::pow(a, b);
    }
}


#ifdef SIMDE_FLOAT16_IS_SCALAR

template<typename U, std::enable_if_t<details::is_powable_num<::nt::float128_t, U>, bool> = true>
NT_ALWAYS_INLINE ::nt::float16_t pow(const ::nt::float16_t& a, const U& b){
    if constexpr (type_traits::is_integral_v<U>){
        return _NT_FLOAT32_TO_FLOAT16_(std::pow(_NT_FLOAT16_TO_FLOAT32_(a), b));
    }else{
        return _NT_FLOAT32_TO_FLOAT16_(std::pow(_NT_FLOAT16_TO_FLOAT32_(a), _NT_FLOAT16_TO_FLOAT32_(b)));
    }
}

#else
template<typename U, std::enable_if_t<details::is_powable_num<::nt::float128_t, U>, bool> = true>
NT_ALWAYS_INLINE ::nt::float16_t pow(const ::nt::float16_t& a, const U& b){
    return half_float::half(half_float::detail::pow(a, b));
}
#endif 

template<typename T, typename U, std::enable_if_t<(type_traits::is_integral_v<U> || type_traits::is_floating_point_v<U>) && !type_traits::is_same_v<bool, U>, bool> = true>
NT_ALWAYS_INLINE ::nt::my_complex<T> pow(nt::my_complex<T> __x, U __y){
	if constexpr (type_traits::is_same_v<T, nt::float16_t>){
        
		return nt::my_complex<T>(std::pow(__x.real(), __y), std::pow(__x.imag(), convert::convert<float16_t>(__y)));
	}else{
		return nt::my_complex<T>(std::pow(__x.real(), T(__y)), std::pow(__x.imag(), T(__y)));
	}
}

template<typename T, std::enable_if_t<type_traits::is_floating_point_v<T>, bool> = true>
NT_ALWAYS_INLINE ::nt::my_complex<T> pow(const ::nt::my_complex<T>& a, const ::nt::my_complex<T>& b){
    return ::nt::my_complex<T>(pow(a.real(), b.real()), pow(a.imag(), b.imag()));
}


template<typename T, typename U, std::enable_if_t<type_traits::is_integral_v<T> && type_traits::is_integral_v<U>, bool> = true>
NT_ALWAYS_INLINE T pow(const T& a, const T& b){
    if constexpr (type_traits::system_int128){
        if constexpr (type_traits::is_same_v<T, ::nt::int128_t>){
            return boost::multiprecision::pow(a, b);
        }else if constexpr (type_traits::is_same_v<T, ::nt::uint128_t>){
            return convert::convert<uint128_t>(std::pow(uint64_t(a), convert::convert<int128_t>(b)));
        }else{
            return std::pow(a, b);
        }
        
    }
    else{
        return std::pow(a, b);
    }
}

#undef NT_FUNCTIONAL_CONVERT_FUNC_CONVERT 
#undef NT_FUNCTIONAL_MASS_INTEGER_CONVERT 
}
