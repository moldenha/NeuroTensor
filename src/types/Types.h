#ifndef _NT_MY_TYPES_H_
#define _NT_MY_TYPES_H_

//#if defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)



#include <complex.h>
#include <ostream>
#include <tuple>
#include <functional> //to make an std::hash path for uint128
/* #include <bfloat16/bfloat16.h> */

#include <simde/simde-f16.h> //using this to convert between float32 and float16
#define _HALF_FLOAT_SUPPORT_ //by default is going to have half float support
#ifdef SIMDE_FLOAT16_IS_SCALAR
namespace nt{
	typedef simde_float16 float16_t;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) simde_float16_to_float32(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f)    simde_float16_from_float32(f)
	inline std::ostream& operator<<(std::ostream& os, const float16_t& val){
		return os << _NT_FLOAT16_TO_FLOAT32_(val);
	}
}
#else
	#include <half/half.hpp>
namespace nt{
	using float16_t = half_float::half;
	#define _NT_FLOAT16_TO_FLOAT32_(f16) float(f16)
	#define _NT_FLOAT32_TO_FLOAT16_(f) float16_t(f)
	//by default has a way to print out the half_float::half
}
#endif



namespace nt{

//basically an std::bit_convert to an int16, all the bits stay the same, just different type name
/* #define _NT_FLOAT16_TO_INT16_(fp) *reinterpret_cast<const int16_t*>(&fp); */
inline int16_t _NT_FLOAT16_TO_INT16_(float16_t fp) noexcept {
	int16_t out = *reinterpret_cast<int16_t*>(&fp);
	return out;
}


/* #ifdef __has_keyword */

/* 	#if __has_keyword(_Float16) */
/* 	#define _HALF_FLOAT_SUPPORT_ */
/* 		using float16_t = _Float16; */
/* 	std::ostream& operator<<(std::ostream& os, const float16_t& val); */

/* 	#elif __has_keyword(__fp16) */
/* 	#define _HALF_FLOAT_SUPPORT_ */
/* 		using float16_t = __fp16; */
/* 	std::ostream& operator<<(std::ostream& os, const float16_t& val); */
/* 	#else */
/* 		#define _HALF_FLOAT_SUPPORT_ */
/* } //nt:: */
/* 		#include <half/half.hpp> */
/* namespace nt{ */
/* 		using float16_t = half_float::half; */
/* 		std::ostream& operator<<(std::ostream& os, const float16_t& val); */
/* 	/1* #define _NO_HALF_FLOAT_SUPPORT_ *1/ */
/* 	#endif */
/* #else */
/* 	#if defined(__STDC_IEC_559__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) */
/* 	// C11 standard with IEC 559 (floating-point arithmetic) support */
/* 	// // You can use _Float16 here */
/* 	#define _HALF_FLOAT_SUPPORT_ */
/* 		using float16_t = _Float16; */
/* 	std::ostream& operator<<(std::ostream& os, const float16_t& val); */
/* 	#elif defined(__GNUC__) && defined(__FP16__) */
/* 	// GCC with support for __fp16 */
/* 	// You can use __fp16 here */
/* 	#define _HALF_FLOAT_SUPPORT_ */
/* 		using float16_t = __fp16; */
/* 	std::ostream& operator<<(std::ostream& os, const float16_t& val); */
/* 	#elif defined(_MSC_VER) && defined(_M_AMD64) && defined(_MSC_VER) && (_MSC_VER >= 1920) */
/* 	// Visual Studio 2019 and later with AMD64 architecture */
/* 	// You can use _Float16 here */
/* 	#define _HALF_FLOAT_SUPPORT_ */
/* 		using float16_t = _Float16; */
/* 	std::ostream& operator<<(std::ostream& os, const float16_t& val); */
/* 	#else */
/* 		#define _HALF_FLOAT_SUPPORT_ */
/* } */
/* 		#include <half/half.hpp> */
/* namespace nt{ */
/* 		using float16_t = half_float::half; */
/* 		std::ostream& operator<<(std::ostream& os, const float16_t& val); */
/* 	// Handle the case when _Float16 is not supported */
/* 	#define _NO_HALF_FLOAT_SUPPORT_ */
/* 	#endif */
/* #endif */


#if defined(__SIZEOF_LONG_DOUBLE__) && __SIZEOF_LONG_DOUBLE__ == 16
#define _128_FLOAT_SUPPORT_
using float128_t = long double;
#elif defined(__has_keyword)
	#if __has_keyword(__float128)
	#define _128_FLOAT_SUPPORT_
		using float128_t = __float128;
		std::ostream& operator<<(std::ostream& os, const float128_t& val);
	#elif __has_keyword(__fp128)
	#define _128_FLOAT_SUPPORT_
		using float128_t = __fp128;
		std::ostream& operator<<(std::ostream& os, const float128_t& val);
	#else
	#define _NO_128_SUPPORT_
	#endif
#else

	#if defined(__STDC_IEC_559__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
		// C11 standard with IEC 559 (floating-point arithmetic) support
		// You can use __float128 here
		#define _128_FLOAT_SUPPORT_
		using float128_t = __float128;
		std::ostream& operator<<(std::ostream& os, const float128_t& val);
	#elif defined(__GNUC__) && defined(__SIZEOF_FLOAT128__)
		// GCC with support for __float128
		// You can use __float128 here
		#define _128_FLOAT_SUPPORT_
		using float128_t = __float128;
		std::ostream& operator<<(std::ostream& os, const float128_t& val);
	#elif defined(_MSC_VER) && defined(_M_AMD64) && defined(_MSC_VER) && (_MSC_VER >= 1920)
		// Visual Studio 2019 and later with AMD64 architecture
		// You can use __float128 here
		#define _128_FLOAT_SUPPORT_
		using float128_t = __float128;
		std::ostream& operator<<(std::ostream& os, const float128_t& val);
	#elif defined(__clang__) && defined(__SIZEOF_FLOAT128__)
		// Clang with support for __float128
		// You can use __float128 here
		#define _128_FLOAT_SUPPORT_
		using float128_t = __float128;
		std::ostream& operator<<(std::ostream& os, const float128_t& val);
	#else
		#define _NO_128_SUPPORT_
		// Handle the case when __float128 is not supported
	#endif
#endif

#ifdef __SIZEOF_INT128__
using uint128_t = __uint128_t;
using int128_t = __int128_t;
std::ostream& operator<<(std::ostream& os, const __int128_t i);
std::ostream& operator<<(std::ostream& os, const __uint128_t i);

#else 

}

//currently has library for uint128 support that is cross platform
//will be adding int128 support that is cross platform soon
#include "uint128_t.h"
namespace nt{
using uint128_t = uint128_t;

}

namespace std{
template<>
struct hash<::nt::uint128_t>{
    std::size_t operator()(const uint128_t& x) const {
        return std::hash<uint64_t>()(static_cast<uint64_t>(x)) ^
               std::hash<uint64_t>()(static_cast<uint64_t>(x >> 64));
    }
};


}

namespace nt{

#endif

}


namespace nt{

/* class complex_32{ */
/* 	float16_t re, im; */
/* 	public: */
/* 		complex_32(const float16_t, const float16_t); */
/* 		complex_32(const std::complex<float16_t>); */
/* 		template<typename T> */
/* 		complex_32& operator=(const T&); */
/* 		complex_32(); */
/* 		float16_t& real(); */
/* 		const float16_t& real() const; */
/* 		float16_t& imag(); */
/* 		const float16_t& imag() const; */
/* 		template<typename T> */
/* 		complex_32& operator*=(const T&); */
/* 		template<typename T> */
/* 		complex_32& operator+=(const T&); */
/* 		template<typename T> */
/* 		complex_32& operator-=(const T&); */
/* 		template<typename T> */
/* 		complex_32& operator/=(const T&); */
		
/* 		template<typename T> */
/* 		complex_32& operator*=(T); */
/* 		template<typename T> */
/* 		complex_32& operator+=(T); */
/* 		template<typename T> */
/* 		complex_32& operator-=(T); */
/* 		template<typename T> */
/* 		complex_32& operator/=(T); */

		
/* 		template<typename T> */
/* 		complex_32 operator*(const T&); */
/* 		template<typename T> */
/* 		complex_32 operator+(const T&); */
/* 		template<typename T> */
/* 		complex_32 operator-(const T&); */
/* 		template<typename T> */
/* 		complex_32 operator/(const T&); */
		
/* 		template<typename T> */
/* 		complex_32 operator*(T); */
/* 		template<typename T> */
/* 		complex_32 operator+(T); */
/* 		template<typename T> */
/* 		complex_32 operator-(T); */
/* 		template<typename T> */
/* 		complex_32 operator/(T); */


/* 		complex_32& operator++(); */
/* 		complex_32 operator++(int); */
/* 		operator std::complex<float16_t>() const; */
/* 		operator complex_64() const; */
/* 		operator complex_128() const; */
/* 		friend std::ostream& operator<<(std::ostream&, const complex_32&); */
/* 		bool operator==(const complex_32&) const; */
/* }; */




/* class complex_64{ */
/* 	float re, im; */
/* 	public: */
/* 		complex_64(const float, const float); */
/* 		complex_64(const std::complex<float>); */
/* 		complex_64(); */
/* 		template<typename T> */
/* 		complex_64& operator=(const T&); */
/* 		float& real(); */
/* 		const float& real() const; */
/* 		float& imag(); */
/* 		const float& imag() const; */
/* 		template<typename T> */
/* 		complex_64& operator*=(const T&); */
/* 		template<typename T> */
/* 		complex_64& operator+=(const T&); */
/* 		template<typename T> */
/* 		complex_64& operator-=(const T&); */
/* 		template<typename T> */
/* 		complex_64& operator/=(const T&); */

/* 		template<typename T> */
/* 		complex_64& operator*=(T); */
/* 		template<typename T> */
/* 		complex_64& operator+=(T); */
/* 		template<typename T> */
/* 		complex_64& operator-=(T); */
/* 		template<typename T> */
/* 		complex_64& operator/=(T); */

/* 		template<typename T> */
/* 		complex_64 operator*(const T&); */
/* 		template<typename T> */
/* 		complex_64 operator+(const T&); */
/* 		template<typename T> */
/* 		complex_64 operator-(const T&); */
/* 		template<typename T> */
/* 		complex_64 operator/(const T&); */
		
/* 		template<typename T> */
/* 		complex_64 operator*(T); */
/* 		template<typename T> */
/* 		complex_64 operator+(T); */
/* 		template<typename T> */
/* 		complex_64 operator-(T); */
/* 		template<typename T> */
/* 		complex_64 operator/(T); */

/* 		complex_64& operator++(); */
/* 		complex_64 operator++(int); */
/* 		operator std::complex<float>() const; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* 		operator complex_32() const; */
/* #endif */
/* 		operator complex_128() const; */
/* 		friend std::ostream& operator<<(std::ostream&, const complex_64&); */
/* 		bool operator==(const complex_64&) const; */

/* }; */



/* class complex_128{ */
/* 	double re, im; */
/* 	public: */
/* 		complex_128(const double, const double); */
/* 		complex_128(const std::complex<double>); */
/* 		template<typename T> */
/* 		complex_128& operator=(const T&); */
/* 		complex_128(); */
/* 		double& real(); */
/* 		const double& real() const; */
/* 		double& imag(); */
/* 		const double& imag() const; */
/* 		template<typename T> */
/* 		complex_128& operator*=(const T&); */
/* 		template<typename T> */
/* 		complex_128& operator+=(const T&); */
/* 		template<typename T> */
/* 		complex_128& operator-=(const T&); */
/* 		template<typename T> */
/* 		complex_128& operator/=(const T&); */

/* 		template<typename T> */
/* 		complex_128 operator*(const T&); */
/* 		template<typename T> */
/* 		complex_128 operator+(const T&); */
/* 		template<typename T> */
/* 		complex_128 operator-(const T&); */
/* 		template<typename T> */
/* 		complex_128 operator/(const T&); */

/* 		template<typename T> */
/* 		complex_128 operator*(T); */
/* 		template<typename T> */
/* 		complex_128 operator+(T); */
/* 		template<typename T> */
/* 		complex_128 operator-(T); */
/* 		template<typename T> */
/* 		complex_128 operator/(T); */

/* 		complex_128& operator++(); */
/* 		complex_128 operator++(int); */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* 		operator complex_32() const; */
/* #endif */
/* 		operator complex_64() const; */
/* 		operator std::complex<double>() const; */
/* 		friend std::ostream& operator<<(std::ostream&, const complex_128&); */
/* 		bool operator==(const complex_128&) const; */

/* }; */

/*

I am trying to do partial specialization in c++ with a typed class function like I have below:

my_complex.h file:

template<typename T>
class my_complex{
	T re, im;
	public:
		my_complex(const T&, const T&);
		template<typename X>
		my_complex<T>& operator+=(const my_complex<T>&)
};

my_complex.cpp file:
template<typename T>
template<typename X>
my_complex<T>& my_complex<T>::operator+=(const my_complex<X>& val){
	re += convert::convert<T>(val.re);
	im += convert::convert<T>(val.im);
	return *this;
}


//partial specialization:
template<typename T>
template<> my_complex<T>& my_complex<T>::operator+=(const my_complex<float>&);
template<typename T>
template my_complex<T>& my_complex<T>::operator+=(const my_complex<double>&);

However, I get the errors:
cannot specialize (with 'template<>') a member of an unspecialized template
and:
expected '<' after 'template'

How can I achieve what I am trying to do with partial specialization?


 */

template<typename T>
class my_complex{
		T re, im;
		template <std::size_t Index, typename U>
		friend inline constexpr U get_complex(const my_complex<U>& obj) noexcept;

		template <std::size_t Index, typename U>
		friend inline constexpr U& get_complex(my_complex<U>& obj) noexcept;

		template <std::size_t Index, typename U>
		friend inline constexpr U&& get_complex(my_complex<U>&& obj) noexcept;

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
#ifdef _HALF_FLOAT_SUPPORT_
using complex_32 = my_complex<float16_t>;
#endif

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
