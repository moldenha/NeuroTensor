#ifndef NT_MATH_FUNCTIONAL_POW_HPP__
#define NT_MATH_FUNCTIONAL_POW_HPP__

#include "utils.h"
#include "general_include.h"

#include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(pow) // <nt/types/float128/math/pow.hpp>

namespace nt::math{

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

template<typename U, std::enable_if_t<details::is_powable_num<float128_t, U>, bool> = true>
NT_ALWAYS_INLINE float128_t pow(const float128_t& a, const U& b){
    if constexpr (type_traits::is_integral_v<U>){
        return f128_ipow(a, convert::convert<int64_t>(b));
    }else{
        return f128_pow(a, b);
    }
}




#ifdef SIMDE_FLOAT16_IS_SCALAR

template<typename U, std::enable_if_t<details::is_powable_num<::nt::float16_t, U>, bool> = true>
NT_ALWAYS_INLINE ::nt::float16_t pow(const ::nt::float16_t& a, const U& b){
    if constexpr (type_traits::is_integral_v<U>){
        return _NT_FLOAT32_TO_FLOAT16_(std::pow(_NT_FLOAT16_TO_FLOAT32_(a), b));
    }else{
        return _NT_FLOAT32_TO_FLOAT16_(std::pow(_NT_FLOAT16_TO_FLOAT32_(a), _NT_FLOAT16_TO_FLOAT32_(b)));
    }
}

#else
template<typename U, std::enable_if_t<details::is_powable_num<::nt::float16_t, U>, bool> = true>
NT_ALWAYS_INLINE ::nt::float16_t pow(const ::nt::float16_t& a, const U& b){
    return half_float::half(half_float::detail::pow(a, b));
}
#endif 

template<typename T, typename U, std::enable_if_t<(type_traits::is_integral_v<U> || type_traits::is_floating_point_v<U>) && !type_traits::is_same_v<bool, U>, bool> = true>
NT_ALWAYS_INLINE ::nt::my_complex<T> pow(nt::my_complex<T> __x, U __y){
	if constexpr (type_traits::is_same_v<T, nt::float16_t>){
		return nt::my_complex<T>(::nt::math::pow(__x.real(), __y), ::nt::math::pow(__x.imag(), __y));
	}
    else{
		return nt::my_complex<T>(::nt::math::pow(__x.real(), __y), ::nt::math::pow(__x.imag(), __y));
	}
}

template<typename T, std::enable_if_t<type_traits::is_floating_point_v<T>, bool> = true>
NT_ALWAYS_INLINE ::nt::my_complex<T> pow(const ::nt::my_complex<T>& a, const ::nt::my_complex<T>& b){
    return ::nt::my_complex<T>(pow(a.real(), b.real()), pow(a.imag(), b.imag()));
}


template<typename T, typename U, std::enable_if_t<type_traits::is_integral_v<T> && type_traits::is_integral_v<U>, bool> = true>
NT_ALWAYS_INLINE T pow(const T& a, const U& b){
    if constexpr (type_traits::system_int128){
#ifdef BOOST_MP_STANDALONE
        if constexpr (type_traits::is_same_v<T, ::nt::int128_t>){
            return boost::multiprecision::pow(a, b);
        }else if constexpr (type_traits::is_same_v<T, ::nt::uint128_t>){
            return convert::convert<uint128_t>(std::pow(uint64_t(a), convert::convert<int128_t>(b)));
        }else{
            return std::pow(a, b);
        }
#else
        if constexpr (type_traits::is_same_v<T, ::nt::uint128_t>){
            return convert::convert<uint128_t>(std::pow(uint64_t(a), convert::convert<int128_t>(b)));
        }else{
            return std::pow(a, b);
        }
#endif
        
    }
    else{
        return std::pow(a, b);
    }
}

template<typename T, typename U, std::enable_if_t<type_traits::is_complex_v<T> && details::is_powable_num<T, U>, bool> = true>
NT_ALWAYS_INLINE T pow(const T& a, const U& b){
    if constexpr (type_traits::is_same_v<T, U>){
        return T(pow(a.real(), b.real()), pow(a.imag(), b.imag()));
    }else{
        return T(pow(a.real(), b), pow(a.imag(), b));
    }
}



}

#endif
