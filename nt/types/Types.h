#ifndef _NT_MY_TYPES_H_
#define _NT_MY_TYPES_H_

//#if defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)
//silence depreciation warnings for certain needed headers
#ifdef _MSC_VER
#ifndef _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#endif

#ifndef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif

#ifndef _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
#define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
#endif

#endif

#include <complex.h>
#include <ostream>
#include <tuple>
#include <functional> //to make an std::hash path for uint128
/* #include <bfloat16/bfloat16.h> */
#include "float16.h"
#include "float128.h"
#include "bit_128_integer.h"



namespace nt{

template<typename T>
class my_complex{
		T re, im;
		template <std::size_t Index, typename U>
		friend inline constexpr U get_complex(const my_complex<U>& obj) noexcept;

		template <std::size_t Index, typename U>
		friend inline constexpr U& get_complex(my_complex<U>& obj) noexcept;

		template <std::size_t Index, typename U>
		friend inline constexpr U&& get_complex(my_complex<U>&& obj) noexcept;
    
        template<typename U>
        friend class my_complex<U>;

	public:
		my_complex(T ele)
			:re(ele), im(ele) {}
		my_complex(const T&, const T&);
		my_complex(const std::complex<T>&);
		my_complex();
		using value_type = T;

		my_complex<T>& operator+=(T);
		my_complex<T>& operator*=(T);
		my_complex<T>& operator-=(T);
		my_complex<T>& operator/=(T);
		// Overload for float
		my_complex<T>& operator+=(const my_complex<float>& val);

		// Overload for double
		my_complex<T>& operator+=(const my_complex<double>& val);

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T>& operator+=(const my_complex<float16_t>& val);
#endif		

		// Overload for float
		my_complex<T>& operator*=(const my_complex<float>& val);

		// Overload for double
		my_complex<T>& operator*=(const my_complex<double>& val);

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T>& operator*=(const my_complex<float16_t>& val);
#endif
		// Overload for float
		my_complex<T>& operator-=(const my_complex<float>& val);

		// Overload for double
		my_complex<T>& operator-=(const my_complex<double>& val);

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T>& operator-=(const my_complex<float16_t>& val);
#endif

		// Overload for float
		my_complex<T>& operator/=(const my_complex<float>& val);

		// Overload for double
		my_complex<T>& operator/=(const my_complex<double>& val);

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T>& operator/=(const my_complex<float16_t>& val);
#endif

		my_complex<T> operator+(T) const;
		my_complex<T> operator*(T) const;
		my_complex<T> operator-(T) const;
		my_complex<T> operator/(T) const;

		// Overload for float
		my_complex<T> operator+(const my_complex<float>& val) const;

		// Overload for double
		my_complex<T> operator+(const my_complex<double>& val) const;

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T> operator+(const my_complex<float16_t>& val) const;
#endif
		// Overload for float
		my_complex<T> operator*(const my_complex<float>& val) const;

		// Overload for double
		my_complex<T> operator*(const my_complex<double>& val) const;

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T> operator*(const my_complex<float16_t>& val) const;
#endif		

		// Overload for float
		my_complex<T> operator-(const my_complex<float>& val) const;

		// Overload for double
		my_complex<T> operator-(const my_complex<double>& val) const;

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T> operator-(const my_complex<float16_t>& val) const;	
#endif

		// Overload for float
		my_complex<T> operator/(const my_complex<float>& val) const;

		// Overload for double
		my_complex<T> operator/(const my_complex<double>& val) const;

#ifdef _HALF_FLOAT_SUPPORT_
		// Overload for float16_t (conditionally included)
		my_complex<T> operator/(const my_complex<float16_t>& val) const;
#endif

		T& real();
		const T& real() const;
		T& imag();
		const T& imag() const;
		my_complex<T>& operator++();
		my_complex<T> operator++(int);
		bool operator==(const my_complex<T>&) const;
		bool operator!=(const my_complex<T>&) const;
		inline friend std::ostream& operator<<(std::ostream& os, const my_complex<T>& c){return os << c.re <<" + "<<c.im<<"i";}
		
		operator std::complex<T>() const;

		template<typename X, std::enable_if_t<!std::is_same_v<T, X>, bool> = true> 
		operator my_complex<X>() const;

		my_complex<T>& operator=(const T&);
		my_complex<T>& operator=(const my_complex<T>&);
		template<typename X, std::enable_if_t<!std::is_same_v<T, X>, bool> = true>
		my_complex<T>& operator=(const my_complex<X>& c);	
		my_complex<T> inverse() const;
		my_complex<T>& inverse_();
		bool operator<(const my_complex<T>&) const;
		bool operator>(const my_complex<T>&) const;
		bool operator<=(const my_complex<T>&) const;
		bool operator>=(const my_complex<T>&) const;
		inline my_complex<T> operator-() const noexcept { return my_complex<T>(-re, -im);}


};


using complex_64 = my_complex<float>;
using complex_128 = my_complex<double>;
using complex_32 = my_complex<float16_t>;

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator/(my_complex<T> comp, U element) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return comp / my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element)));
	}else{
		return comp / my_complex<T>(T(element));
	}
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator/(U element, my_complex<T> comp) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element))) / comp;
	}else{
		return my_complex<T>(T(element)) / comp;
	}
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator*(my_complex<T> comp, U element) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return comp * my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element)));
	}else{
		return comp * my_complex<T>(T(element));
	}
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator*(U element, my_complex<T> comp) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element))) * comp;
	}else{
		return my_complex<T>(T(element)) * comp;
	}
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator+(my_complex<T> comp, U element) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return comp + my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element)));
	}else{
		return comp + my_complex<T>(T(element));
	}
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator+(U element, my_complex<T> comp) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element))) + comp;
	}else{
		return my_complex<T>(T(element)) + comp;
	}
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator-(my_complex<T> comp, U element) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return comp - my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element)));
	}else{
		return comp - my_complex<T>(T(element));
	}
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline my_complex<T> operator-(U element, my_complex<T> comp) noexcept {
	if constexpr (std::is_same_v<T, float16_t>){
		return my_complex<T>(_NT_FLOAT32_TO_FLOAT16_(float(element))) - comp;
	}else{
		return my_complex<T>(T(element)) - comp;
	}
}


template <std::size_t Index, typename T>
inline constexpr T get_complex(const my_complex<T>& c) noexcept {
    static_assert(Index < 2, "Index out of bounds for nt::my_complex");
    if constexpr (Index == 0) {
        return c.re;  // Access allowed due to 'friend' declaration
    } else {
        return c.im;
    }
}

template <std::size_t Index, typename T>
inline constexpr T& get_complex(my_complex<T>& c) noexcept {
    static_assert(Index < 2, "Index out of bounds for nt::my_complex");
    if constexpr (Index == 0) {
        return c.re;  // Access allowed due to 'friend' declaration
    } else {
        return c.im;
    }
}

template <std::size_t Index, typename T>
inline constexpr T&& get_complex(my_complex<T>&& c) noexcept {
    static_assert(Index < 2, "Index out of bounds for nt::my_complex");
    if constexpr (Index == 0) {
        return std::move(c.re);  // Access allowed due to 'friend' declaration
    } else {
        return std::move(c.im);
   }
}


struct uint_bool_t{
	uint8_t value : 1;
	uint_bool_t();
	uint_bool_t(const bool& val);
	uint_bool_t(const uint_bool_t& val);
	uint_bool_t(uint_bool_t&& val);
	inline uint_bool_t& operator=(const bool& val){value = val ? 1 : 0; return *this;}
	inline uint_bool_t& operator=(const uint8_t &val){value = val > 0 ? 1 : 0; return *this;}
	inline uint_bool_t& operator=(const uint_bool_t &val){value = val.value; return *this;}
	inline uint_bool_t& operator=(uint_bool_t&& val){value = val.value; return *this;}
    inline operator bool() const {return value == 1;}
	friend bool operator==(const uint_bool_t& a, const uint_bool_t& b);	
	friend bool operator==(const bool& a, const uint_bool_t& b);	
	friend bool operator==(const uint_bool_t& a, const bool& b);	
};


}

// Enable ADL for `get`
namespace std {
// Overloads of std::get for nt::my_complex
template <std::size_t Index, typename T>
inline constexpr T get(const nt::my_complex<T>& c) noexcept {
	return nt::get_complex<Index>(c);
}

template <std::size_t Index, typename T>
inline constexpr T& get(nt::my_complex<T>& c) noexcept {
	return nt::get_complex<Index>(c);
}

template <std::size_t Index, typename T>
inline constexpr T&& get(nt::my_complex<T>&& c) noexcept {
	return nt::get_complex<Index>(c);
}


}  // namespace std

// Specialize `std::tuple_size` and `std::tuple_element` to make std::complex tuple-like
namespace std {
    template <typename T>
    struct tuple_size<nt::my_complex<T>> : std::integral_constant<size_t, 2> {};

    template <size_t Index, typename T>
    struct tuple_element<Index, nt::my_complex<T>> {
        using type = T;
    };
}


#define _NT_DEFINE_STL_FUNC_FP16_ROUTE_(route)\
	inline ::nt::float16_t route(::nt::float16_t num) { return _NT_FLOAT32_TO_FLOAT16_(route(_NT_FLOAT16_TO_FLOAT32_(num)));}

#define _NT_DEFINE_STL_FUNC_CFP16_ROUTE_(route)\
	inline ::nt::complex_32 route(::nt::complex_32 num) { \
		return ::nt::complex_32( _NT_FLOAT32_TO_FLOAT16_(route(_NT_FLOAT16_TO_FLOAT32_(std::get<0>(num)))),\
					_NT_FLOAT32_TO_FLOAT16_(route(_NT_FLOAT16_TO_FLOAT32_(std::get<1>(num)))));\
	}

namespace std{
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(exp)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(exp)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(abs)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(abs)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(sqrt)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(sqrt)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(log)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(log)

_NT_DEFINE_STL_FUNC_FP16_ROUTE_(tanh)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(tanh)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(tan)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(tan)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(atan)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(atan)

_NT_DEFINE_STL_FUNC_FP16_ROUTE_(sinh)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(sinh)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(sin)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(sin)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(asin)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(asin)

_NT_DEFINE_STL_FUNC_FP16_ROUTE_(cosh)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(cosh)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(cos)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(cos)
_NT_DEFINE_STL_FUNC_FP16_ROUTE_(acos)
_NT_DEFINE_STL_FUNC_CFP16_ROUTE_(acos)



inline ::nt::float16_t pow(::nt::float16_t a, ::nt::float16_t b){
    return _NT_FLOAT32_TO_FLOAT16_(std::pow(_NT_FLOAT16_TO_FLOAT32_(a), _NT_FLOAT16_TO_FLOAT32_(b)));
}

template<typename T>
inline ::nt::my_complex<T> pow(::nt::my_complex<T> __x, ::nt::my_complex<T> __y){
	return nt::my_complex<T>(std::pow(std::get<0>(__x), std::get<0>(__y)), std::pow(std::get<1>(__x), std::get<1>(__y)));
}

template<typename T, typename U, std::enable_if_t<(std::is_integral<U>::value || std::is_floating_point<U>::value) && !std::is_same_v<bool, U>, bool> = true>
inline nt::my_complex<T> pow(nt::my_complex<T> __x, U __y){
	if constexpr (std::is_same_v<T, nt::float16_t>){
		return nt::my_complex<T>(pow(get<0>(__x), _NT_FLOAT32_TO_FLOAT16_(float(__y))), pow(get<1>(__x), _NT_FLOAT32_TO_FLOAT16_(float(__y))));
	}else{
		return nt::my_complex<T>(pow(get<0>(__x), T(__y)), pow(get<1>(__x), T(__y)));
	}
}


}


#endif // _NT_MY_TYPES_H_
