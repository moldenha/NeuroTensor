#ifndef _MY_TYPES_H_
#define _MY_TYPES_H_

//#if defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)



#include <complex.h>
#include <ostream>
/* #include <bfloat16/bfloat16.h> */

namespace nt{


#ifdef __has_keyword

	#if __has_keyword(_Float16)
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = _Float16;
	std::ostream& operator<<(std::ostream& os, const float16_t& val);

	#elif __has_keyword(__fp16)
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = __fp16;
	std::ostream& operator<<(std::ostream& os, const float16_t& val);
	#else
	#define _NO_HALF_FLOAT_SUPPORT_
	#endif
#else
	#if defined(__STDC_IEC_559__) && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
	// C11 standard with IEC 559 (floating-point arithmetic) support
	// // You can use _Float16 here
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = _Float16;
	std::ostream& operator<<(std::ostream& os, const float16_t& val);
	#elif defined(__GNUC__) && defined(__FP16__)
	// GCC with support for __fp16
	// You can use __fp16 here
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = __fp16;
	std::ostream& operator<<(std::ostream& os, const float16_t& val);
	#elif defined(_MSC_VER) && defined(_M_AMD64) && defined(_MSC_VER) && (_MSC_VER >= 1920)
	// Visual Studio 2019 and later with AMD64 architecture
	// You can use _Float16 here
	#define _HALF_FLOAT_SUPPORT_
		using float16_t = _Float16;
	std::ostream& operator<<(std::ostream& os, const float16_t& val);
	
	#else
	// Handle the case when _Float16 is not supported
	#define _NO_HALF_FLOAT_SUPPORT_
	#endif
#endif


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
	public:
		my_complex(const T&, const T&);
		my_complex(const std::complex<T>&);
		my_complex();

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


};


using complex_64 = my_complex<float>;
using complex_128 = my_complex<double>;
#ifdef _HALF_FLOAT_SUPPORT_
using complex_32 = my_complex<float16_t>;
#endif


struct uint_bool_t{
	unsigned value : 1;
	uint_bool_t();
	uint_bool_t(const bool& val);
	uint_bool_t(const uint_bool_t& val);
	uint_bool_t(uint_bool_t&& val);
	inline uint_bool_t& operator=(const bool& val){value = val ? 1 : 0; return *this;}
	inline uint_bool_t& operator=(const uint8_t &val){value = val > 0 ? 1 : 0; return *this;}
	inline uint_bool_t& operator=(const uint_bool_t &val){value = val.value; return *this;}
	inline uint_bool_t& operator=(uint_bool_t&& val){value = val.value; return *this;}
	friend bool operator==(const uint_bool_t& a, const uint_bool_t& b);	
	friend bool operator==(const bool& a, const uint_bool_t& b);	
	friend bool operator==(const uint_bool_t& a, const bool& b);	
};


}


#endif
