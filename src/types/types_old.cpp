#include "Types.h"
#include <ostream>
#include <sys/_types/_int64_t.h>
#include <type_traits>

namespace nt{
#ifdef _HALF_FLOAT_SUPPORT_

inline std::ostream& operator<<(std::ostream& os, const float16_t& val){
	os << static_cast<float>(val);
	return os;
}

complex_32::complex_32()
	:re(0), im(0)
{}

complex_32::complex_32(const float16_t re, const float16_t im)
	:re(re), im(im)
{}

complex_32::complex_32(const std::complex<float16_t> a)
	:re(a.real()), im(a.imag())
{}

inline complex_32& complex_32::operator++(){
	++re;
	++im;
	return *this;
}

inline complex_32 complex_32::operator++(int){
	complex_32 tmp = *this;
	++(*this);
	return tmp;
}


template<typename T>
inline complex_32& complex_32::operator*=(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re *= a.re;
		im *= a.im;
		return *this;
	}
	else{
		re *= a;
		return *this;
	}
}


template<typename T>
inline complex_32& complex_32::operator+=(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re += a.re;
		im += a.im;
		return *this;
	}
	else{
		re += a;
		return *this;
	}
}

template<typename T>
inline complex_32& complex_32::operator-=(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re -= a.re;
		im -= a.im;
		return *this;
	}
	else{
		re -= a;
		return *this;
	}
}

template<typename T>
inline complex_32& complex_32::operator/=(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re /= a.re;
		im /= a.im;
		return *this;
	}
	else{
		re /= a;
		return *this;
	}
}

template complex_32& complex_32::operator*=(complex_32 &a);
template complex_32& complex_32::operator*=(complex_64 &a);
template complex_32& complex_32::operator*=(complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator*=(float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator*=(int128_t &a);
template complex_32& complex_32::operator*=(uint128_t &a);
#endif
template complex_32& complex_32::operator*=(double &a);
template complex_32& complex_32::operator*=(float &a);
template complex_32& complex_32::operator*=(float16_t &a);
template complex_32& complex_32::operator*=(int64_t &a);
template complex_32& complex_32::operator*=(uint32_t &a);
template complex_32& complex_32::operator*=(int32_t &a);
template complex_32& complex_32::operator*=(uint16_t &a);
template complex_32& complex_32::operator*=(int16_t &a);
template complex_32& complex_32::operator*=(uint8_t &a);
template complex_32& complex_32::operator*=(int8_t &a);

template complex_32& complex_32::operator+=(complex_32 &a);
template complex_32& complex_32::operator+=(complex_64 &a);
template complex_32& complex_32::operator+=(complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator+=(float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator+=(int128_t &a);
template complex_32& complex_32::operator+=(uint128_t &a);
#endif
template complex_32& complex_32::operator+=(double &a);
template complex_32& complex_32::operator+=(float &a);
template complex_32& complex_32::operator+=(float16_t &a);
template complex_32& complex_32::operator+=(int64_t &a);
template complex_32& complex_32::operator+=(uint32_t &a);
template complex_32& complex_32::operator+=(int32_t &a);
template complex_32& complex_32::operator+=(uint16_t &a);
template complex_32& complex_32::operator+=(int16_t &a);
template complex_32& complex_32::operator+=(uint8_t &a);
template complex_32& complex_32::operator+=(int8_t &a);

template complex_32& complex_32::operator-=(complex_32 &a);
template complex_32& complex_32::operator-=(complex_64 &a);
template complex_32& complex_32::operator-=(complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator-=(float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator-=(int128_t &a);
template complex_32& complex_32::operator-=(uint128_t &a);
#endif
template complex_32& complex_32::operator-=(double &a);
template complex_32& complex_32::operator-=(float &a);
template complex_32& complex_32::operator-=(float16_t &a);
template complex_32& complex_32::operator-=(int64_t &a);
template complex_32& complex_32::operator-=(uint32_t &a);
template complex_32& complex_32::operator-=(int32_t &a);
template complex_32& complex_32::operator-=(uint16_t &a);
template complex_32& complex_32::operator-=(int16_t &a);
template complex_32& complex_32::operator-=(uint8_t &a);
template complex_32& complex_32::operator-=(int8_t &a);

template complex_32& complex_32::operator/=(complex_32 &a);
template complex_32& complex_32::operator/=(complex_64 &a);
template complex_32& complex_32::operator/=(complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator/=(float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator/=(int128_t &a);
template complex_32& complex_32::operator/=(uint128_t &a);
#endif
template complex_32& complex_32::operator/=(double &a);
template complex_32& complex_32::operator/=(float &a);
template complex_32& complex_32::operator/=(float16_t &a);
template complex_32& complex_32::operator/=(int64_t &a);
template complex_32& complex_32::operator/=(uint32_t &a);
template complex_32& complex_32::operator/=(int32_t &a);
template complex_32& complex_32::operator/=(uint16_t &a);
template complex_32& complex_32::operator/=(int16_t &a);
template complex_32& complex_32::operator/=(uint8_t &a);
template complex_32& complex_32::operator/=(int8_t &a);

template<typename T>
complex_32& complex_32::operator*=(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re *= v.re;
		im *= v.im;
		return *this;
	}
	else{
		re *= v;
		return *this;
	}
}
template<typename T>
complex_32& complex_32::operator+=(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re += v.re;
		im += v.im;
		return *this;
	}
	else{
		re += v;
		return *this;
	}
}
template<typename T>
complex_32& complex_32::operator-=(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re -= v.re;
		im -= v.im;
		return *this;
	}
	else{
		re -= v;
		return *this;
	}

}
template<typename T>
complex_32& complex_32::operator/=(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re /= v.re;
		im /= v.im;
		return *this;
	}
	else{
		re /= v;
		return *this;
	}

}


template complex_32& complex_32::operator*=(complex_32 v);
template complex_32& complex_32::operator*=(complex_64 v);
template complex_32& complex_32::operator*=(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator*=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator*=(int128_t v);
template complex_32& complex_32::operator*=(uint128_t v);
#endif
template complex_32& complex_32::operator*=(double v);
template complex_32& complex_32::operator*=(float v);
template complex_32& complex_32::operator*=(float16_t v);
template complex_32& complex_32::operator*=(int64_t v);
template complex_32& complex_32::operator*=(uint32_t v);
template complex_32& complex_32::operator*=(int32_t v);
template complex_32& complex_32::operator*=(uint16_t v);
template complex_32& complex_32::operator*=(int16_t v);
template complex_32& complex_32::operator*=(uint8_t v);
template complex_32& complex_32::operator*=(int8_t v);

template complex_32& complex_32::operator+=(complex_32 v);
template complex_32& complex_32::operator+=(complex_64 v);
template complex_32& complex_32::operator+=(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator+=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator+=(int128_t v);
template complex_32& complex_32::operator+=(uint128_t v);
#endif
template complex_32& complex_32::operator+=(double v);
template complex_32& complex_32::operator+=(float v);
template complex_32& complex_32::operator+=(float16_t v);
template complex_32& complex_32::operator+=(int64_t v);
template complex_32& complex_32::operator+=(uint32_t v);
template complex_32& complex_32::operator+=(int32_t v);
template complex_32& complex_32::operator+=(uint16_t v);
template complex_32& complex_32::operator+=(int16_t v);
template complex_32& complex_32::operator+=(uint8_t v);
template complex_32& complex_32::operator+=(int8_t v);

template complex_32& complex_32::operator-=(complex_32 v);
template complex_32& complex_32::operator-=(complex_64 v);
template complex_32& complex_32::operator-=(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator-=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator-=(int128_t v);
template complex_32& complex_32::operator-=(uint128_t v);
#endif
template complex_32& complex_32::operator-=(double v);
template complex_32& complex_32::operator-=(float v);
template complex_32& complex_32::operator-=(float16_t v);
template complex_32& complex_32::operator-=(int64_t v);
template complex_32& complex_32::operator-=(uint32_t v);
template complex_32& complex_32::operator-=(int32_t v);
template complex_32& complex_32::operator-=(uint16_t v);
template complex_32& complex_32::operator-=(int16_t v);
template complex_32& complex_32::operator-=(uint8_t v);
template complex_32& complex_32::operator-=(int8_t v);

template complex_32& complex_32::operator/=(complex_32 v);
template complex_32& complex_32::operator/=(complex_64 v);
template complex_32& complex_32::operator/=(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator/=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator/=(int128_t v);
template complex_32& complex_32::operator/=(uint128_t v);
#endif
template complex_32& complex_32::operator/=(double v);
template complex_32& complex_32::operator/=(float v);
template complex_32& complex_32::operator/=(float16_t v);
template complex_32& complex_32::operator/=(int64_t v);
template complex_32& complex_32::operator/=(uint32_t v);
template complex_32& complex_32::operator/=(int32_t v);
template complex_32& complex_32::operator/=(uint16_t v);
template complex_32& complex_32::operator/=(int16_t v);
template complex_32& complex_32::operator/=(uint8_t v);
template complex_32& complex_32::operator/=(int8_t v);

template<typename T>
inline complex_32 complex_32::operator*(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re * a.re, im * a.im);
	}
	else{
		return complex_32(re * a, im);
	}
}

template<typename T>
inline complex_32 complex_32::operator+(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re + a.re, im + a.im);
	}
	else{
		return complex_32(re + a, im);
	}
}

template<typename T>
inline complex_32 complex_32::operator-(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re - a.re, im - a.im);
	}
	else{
		return complex_32(re - a, im);
	}
}
template<typename T>
inline complex_32 complex_32::operator/(const T &a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re / a.re, im / a.im);
	}
	else{
		return complex_32(re / a, im);
	}
}


template complex_32 complex_32::operator*(const complex_32 &a);
template complex_32 complex_32::operator*(const complex_64 &a);
template complex_32 complex_32::operator*(const complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator*(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator*(const int128_t &a);
template complex_32 complex_32::operator*(const uint128_t &a);
#endif
template complex_32 complex_32::operator*(const double &a);
template complex_32 complex_32::operator*(const float &a);
template complex_32 complex_32::operator*(const float16_t &a);
template complex_32 complex_32::operator*(const int64_t &a);
template complex_32 complex_32::operator*(const uint32_t &a);
template complex_32 complex_32::operator*(const int32_t &a);
template complex_32 complex_32::operator*(const uint16_t &a);
template complex_32 complex_32::operator*(const int16_t &a);
template complex_32 complex_32::operator*(const uint8_t &a);
template complex_32 complex_32::operator*(const int8_t &a);

template complex_32 complex_32::operator+(const complex_32 &a);
template complex_32 complex_32::operator+(const complex_64 &a);
template complex_32 complex_32::operator+(const complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator+(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator+(const int128_t &a);
template complex_32 complex_32::operator+(const uint128_t &a);
#endif
template complex_32 complex_32::operator+(const double &a);
template complex_32 complex_32::operator+(const float &a);
template complex_32 complex_32::operator+(const float16_t &a);
template complex_32 complex_32::operator+(const int64_t &a);
template complex_32 complex_32::operator+(const uint32_t &a);
template complex_32 complex_32::operator+(const int32_t &a);
template complex_32 complex_32::operator+(const uint16_t &a);
template complex_32 complex_32::operator+(const int16_t &a);
template complex_32 complex_32::operator+(const uint8_t &a);
template complex_32 complex_32::operator+(const int8_t &a);

template complex_32 complex_32::operator-(const complex_32 &a);
template complex_32 complex_32::operator-(const complex_64 &a);
template complex_32 complex_32::operator-(const complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator-(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator-(const int128_t &a);
template complex_32 complex_32::operator-(const uint128_t &a);
#endif
template complex_32 complex_32::operator-(const double &a);
template complex_32 complex_32::operator-(const float &a);
template complex_32 complex_32::operator-(const float16_t &a);
template complex_32 complex_32::operator-(const int64_t &a);
template complex_32 complex_32::operator-(const uint32_t &a);
template complex_32 complex_32::operator-(const int32_t &a);
template complex_32 complex_32::operator-(const uint16_t &a);
template complex_32 complex_32::operator-(const int16_t &a);
template complex_32 complex_32::operator-(const uint8_t &a);
template complex_32 complex_32::operator-(const int8_t &a);

template complex_32 complex_32::operator/(const complex_32 &a);
template complex_32 complex_32::operator/(const complex_64 &a);
template complex_32 complex_32::operator/(const complex_128 &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator/(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator/(const int128_t &a);
template complex_32 complex_32::operator/(const uint128_t &a);
#endif
template complex_32 complex_32::operator/(const double &a);
template complex_32 complex_32::operator/(const float &a);
template complex_32 complex_32::operator/(const float16_t &a);
template complex_32 complex_32::operator/(const int64_t &a);
template complex_32 complex_32::operator/(const uint32_t &a);
template complex_32 complex_32::operator/(const int32_t &a);
template complex_32 complex_32::operator/(const uint16_t &a);
template complex_32 complex_32::operator/(const int16_t &a);
template complex_32 complex_32::operator/(const uint8_t &a);
template complex_32 complex_32::operator/(const int8_t &a);

template<typename T>
complex_32 complex_32::operator*(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re * v.re, im * v.im);
	}
	else{
		return complex_32(re * v, im);
	}
}
template<typename T>
complex_32 complex_32::operator+(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re + v.re, im + v.im);
	}
	else{
		return complex_32(re + v, im);
	}
}
template<typename T>
complex_32 complex_32::operator-(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re - v.re, im - v.im);
	}
	else{
		return complex_32(re - v, im);
	}
}
template<typename T>
complex_32 complex_32::operator/(T v){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_32(re / v.re, im / v.im);
	}
	else{
		return complex_32(re / v, im);
	}
}


template complex_32 complex_32::operator*(complex_32 v);
template complex_32 complex_32::operator*(complex_64 v);
template complex_32 complex_32::operator*(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator*(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator*(int128_t v);
template complex_32 complex_32::operator*(uint128_t v);
#endif
template complex_32 complex_32::operator*(double v);
template complex_32 complex_32::operator*(float v);
template complex_32 complex_32::operator*(float16_t v);
template complex_32 complex_32::operator*(int64_t v);
template complex_32 complex_32::operator*(uint32_t v);
template complex_32 complex_32::operator*(int32_t v);
template complex_32 complex_32::operator*(uint16_t v);
template complex_32 complex_32::operator*(int16_t v);
template complex_32 complex_32::operator*(uint8_t v);
template complex_32 complex_32::operator*(int8_t v);

template complex_32 complex_32::operator+(complex_32 v);
template complex_32 complex_32::operator+(complex_64 v);
template complex_32 complex_32::operator+(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator+(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator+(int128_t v);
template complex_32 complex_32::operator+(uint128_t v);
#endif
template complex_32 complex_32::operator+(double v);
template complex_32 complex_32::operator+(float v);
template complex_32 complex_32::operator+(float16_t v);
template complex_32 complex_32::operator+(int64_t v);
template complex_32 complex_32::operator+(uint32_t v);
template complex_32 complex_32::operator+(int32_t v);
template complex_32 complex_32::operator+(uint16_t v);
template complex_32 complex_32::operator+(int16_t v);
template complex_32 complex_32::operator+(uint8_t v);
template complex_32 complex_32::operator+(int8_t v);

template complex_32 complex_32::operator-(complex_32 v);
template complex_32 complex_32::operator-(complex_64 v);
template complex_32 complex_32::operator-(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator-(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator-(int128_t v);
template complex_32 complex_32::operator-(uint128_t v);
#endif
template complex_32 complex_32::operator-(double v);
template complex_32 complex_32::operator-(float v);
template complex_32 complex_32::operator-(float16_t v);
template complex_32 complex_32::operator-(int64_t v);
template complex_32 complex_32::operator-(uint32_t v);
template complex_32 complex_32::operator-(int32_t v);
template complex_32 complex_32::operator-(uint16_t v);
template complex_32 complex_32::operator-(int16_t v);
template complex_32 complex_32::operator-(uint8_t v);
template complex_32 complex_32::operator-(int8_t v);

template complex_32 complex_32::operator/(complex_32 v);
template complex_32 complex_32::operator/(complex_64 v);
template complex_32 complex_32::operator/(complex_128 v);
#ifdef _128_FLOAT_SUPPORT_
template complex_32 complex_32::operator/(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_32 complex_32::operator/(int128_t v);
template complex_32 complex_32::operator/(uint128_t v);
#endif
template complex_32 complex_32::operator/(double v);
template complex_32 complex_32::operator/(float v);
template complex_32 complex_32::operator/(float16_t v);
template complex_32 complex_32::operator/(int64_t v);
template complex_32 complex_32::operator/(uint32_t v);
template complex_32 complex_32::operator/(int32_t v);
template complex_32 complex_32::operator/(uint16_t v);
template complex_32 complex_32::operator/(int16_t v);
template complex_32 complex_32::operator/(uint8_t v);
template complex_32 complex_32::operator/(int8_t v);


inline float16_t& complex_32::real(){return re;}
inline const float16_t& complex_32::real() const{return re;}
inline float16_t& complex_32::imag(){return im;}
inline const float16_t& complex_32::imag() const{return im;}

inline complex_32::operator std::complex<float16_t>() const{return std::complex<float16_t>(re, im);}
inline complex_32::operator complex_128() const {return complex_128(static_cast<double>(re), static_cast<double>(im));}
inline complex_32::operator complex_64() const {return complex_64(static_cast<float>(re), static_cast<float>(im));}
inline bool complex_32::operator==(const complex_32& a) const{
	return a.re == re && a.im == im;
}

template<typename T>
inline complex_32& complex_32::operator=(const T& a){
	if constexpr(std::is_same_v<T, complex_32> || std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re = a.re;
		im = a.im;
		return *this;
	}
	else if constexpr(std::is_same_v<T, std::complex<float16_t>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>){
		re = a.real();
		im = a.imag();
		return *this;
	} 
	else{
		re = a;
		return *this;
	}
}


template complex_32& complex_32::operator=(const complex_64 &a);
template complex_32& complex_32::operator=(const complex_128 &a);
template complex_32& complex_32::operator=(const complex_32 &a);
template complex_32& complex_32::operator=(const std::complex<double>&a);
template complex_32& complex_32::operator=(const std::complex<float> &a);
template complex_32& complex_32::operator=(const std::complex<float16_t> &a);
#ifdef _128_FLOAT_SUPPORT_
template complex_32& complex_32::operator=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_32& complex_32::operator=(const int128_t &a);
template complex_32& complex_32::operator=(const uint128_t &a);
#endif
template complex_32& complex_32::operator=(const double &a);
template complex_32& complex_32::operator=(const float &a);
template complex_32& complex_32::operator=(const float16_t &a);
template complex_32& complex_32::operator=(const int64_t &a);
template complex_32& complex_32::operator=(const uint32_t &a);
template complex_32& complex_32::operator=(const int32_t &a);
template complex_32& complex_32::operator=(const uint16_t &a);
template complex_32& complex_32::operator=(const int16_t &a);
template complex_32& complex_32::operator=(const uint8_t &a);
template complex_32& complex_32::operator=(const int8_t &a);



#endif

#ifdef __SIZEOF_INT128__ 
std::ostream& operator<<(std::ostream& os, const int128_t i){
  std::ostream::sentry s(os);
  if (s) {
    nt::uint128_t tmp = i < 0 ? -i : i;
    char buffer[128];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[tmp % 10];
      tmp /= 10;
    } while (tmp != 0);
    if (i < 0) {
      --d;
      *d = '-';
    }
    int len = std::end(buffer) - d;
    if (os.rdbuf()->sputn(d, len) != len) {
      os.setstate(std::ios_base::badbit);
    }
  }
  return os;
}
std::ostream& operator<<(std::ostream& os, const uint128_t i){
  std::ostream::sentry s(os);
  if (s) {
    nt::uint128_t tmp = i;
    char buffer[128];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[tmp % 10];
      tmp /= 10;
    } while (tmp != 0);
    int len = std::end(buffer) - d;
    if (os.rdbuf()->sputn(d, len) != len) {
      os.setstate(std::ios_base::badbit);
    }
  }
  return os;

}
#endif




uint_bool_t::uint_bool_t():value(0){}
uint_bool_t::uint_bool_t(const bool& val):value(val ? 1 : 0){}
uint_bool_t::uint_bool_t(const uint_bool_t& val):value(val.value){}
uint_bool_t::uint_bool_t(uint_bool_t&& val):value(val.value){}
inline uint_bool_t& uint_bool_t::operator=(const bool& val){value = val ? 1 : 0; return *this;}
inline uint_bool_t& uint_bool_t::operator=(const uint8_t &val){value = val > 0 ? 1 : 0; return *this;}
inline uint_bool_t& uint_bool_t::operator=(const uint_bool_t &val){value = val.value; return *this;}
inline uint_bool_t& uint_bool_t::operator=(uint_bool_t&& val){value = val.value; return *this;}


inline bool operator==(const uint_bool_t& a, const uint_bool_t& b){return a.value == b.value;}
inline bool operator==(const uint_bool_t& a, const bool& b){return (a.value == 1 && b) || (a.value == 0 && !b);}
inline bool operator==(const bool& a, const uint_bool_t& b){return (b.value == 1 && a) || (b.value == 0 && !a);}


complex_64::complex_64()
	:re(0), im(0)
{}

complex_64::complex_64(const float re, const float im)
	:re(re), im(im)
{}

complex_64::complex_64(const std::complex<float> a)
	:re(a.real()), im(a.imag())
{}


inline complex_64& complex_64::operator++(){
	++re;
	++im;
	return *this;
}

inline complex_64 complex_64::operator++(int){
	complex_64 tmp = *this;
	++(*this);
	return tmp;
}

template<typename T>
inline complex_64& complex_64::operator*=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re *= a.re;
		im *= a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re *= a.re;
		im *= a.im;
		return *this;
	}
#endif
	else{
		re *= a;
		return *this;
	}
}


template<typename T>
inline complex_64& complex_64::operator+=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re += a.re;
		im += a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re += a.re;
		im += a.im;
		return *this;	
	}
#endif
	else{
		re += a;
		return *this;
	}
}

template<typename T>
inline complex_64& complex_64::operator-=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re -= a.re;
		im -= a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re -= a.re;
		im -= a.im;
		return *this;	
	}
#endif

	else{
		re -= a;
		return *this;
	}
}

template<typename T>
inline complex_64& complex_64::operator/=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re /= a.re;
		im /= a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re /= a.re;
		im /= a.im;
		return *this;	
	}
#endif

	else{
		re /= a;
		return *this;
	}
}

template complex_64& complex_64::operator*=(const complex_64 &a);
template complex_64& complex_64::operator*=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator*=(const complex_32 &a);
template complex_64& complex_64::operator*=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator*=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator*=(const int128_t &a);
template complex_64& complex_64::operator*=(const uint128_t &a);
#endif
template complex_64& complex_64::operator*=(const double &a);
template complex_64& complex_64::operator*=(const float &a);
template complex_64& complex_64::operator*=(const int64_t &a);
template complex_64& complex_64::operator*=(const uint32_t &a);
template complex_64& complex_64::operator*=(const int32_t &a);
template complex_64& complex_64::operator*=(const uint16_t &a);
template complex_64& complex_64::operator*=(const int16_t &a);
template complex_64& complex_64::operator*=(const uint8_t &a);
template complex_64& complex_64::operator*=(const int8_t &a);

template complex_64& complex_64::operator+=(const complex_64 &a);
template complex_64& complex_64::operator+=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator+=(const complex_32 &a);
template complex_64& complex_64::operator+=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator+=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator+=(const int128_t &a);
template complex_64& complex_64::operator+=(const uint128_t &a);
#endif
template complex_64& complex_64::operator+=(const double &a);
template complex_64& complex_64::operator+=(const float &a);
template complex_64& complex_64::operator+=(const int64_t &a);
template complex_64& complex_64::operator+=(const uint32_t &a);
template complex_64& complex_64::operator+=(const int32_t &a);
template complex_64& complex_64::operator+=(const uint16_t &a);
template complex_64& complex_64::operator+=(const int16_t &a);
template complex_64& complex_64::operator+=(const uint8_t &a);
template complex_64& complex_64::operator+=(const int8_t &a);

template complex_64& complex_64::operator-=(const complex_64 &a);
template complex_64& complex_64::operator-=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator-=(const complex_32 &a);
template complex_64& complex_64::operator-=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator-=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator-=(const int128_t &a);
template complex_64& complex_64::operator-=(const uint128_t &a);
#endif
template complex_64& complex_64::operator-=(const double &a);
template complex_64& complex_64::operator-=(const float &a);
template complex_64& complex_64::operator-=(const int64_t &a);
template complex_64& complex_64::operator-=(const uint32_t &a);
template complex_64& complex_64::operator-=(const int32_t &a);
template complex_64& complex_64::operator-=(const uint16_t &a);
template complex_64& complex_64::operator-=(const int16_t &a);
template complex_64& complex_64::operator-=(const uint8_t &a);
template complex_64& complex_64::operator-=(const int8_t &a);

template complex_64& complex_64::operator/=(const complex_64 &a);
template complex_64& complex_64::operator/=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator/=(const complex_32 &a);
template complex_64& complex_64::operator/=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator/=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator/=(const int128_t &a);
template complex_64& complex_64::operator/=(const uint128_t &a);
#endif
template complex_64& complex_64::operator/=(const double &a);
template complex_64& complex_64::operator/=(const float &a);
template complex_64& complex_64::operator/=(const int64_t &a);
template complex_64& complex_64::operator/=(const uint32_t &a);
template complex_64& complex_64::operator/=(const int32_t &a);
template complex_64& complex_64::operator/=(const uint16_t &a);
template complex_64& complex_64::operator/=(const int16_t &a);
template complex_64& complex_64::operator/=(const uint8_t &a);
template complex_64& complex_64::operator/=(const int8_t &a);


template<typename T>
inline complex_64& complex_64::operator*=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re *= v.re;
		im *= v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re *= v.re;
		im *= v.im;
		return *this;
	
	}
#endif
	else{
		re *= v;
		return *this;
	}}


template<typename T>
inline complex_64& complex_64::operator+=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re += v.re;
		im += v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re += v.re;
		im += v.im;
		return *this;
	
	}
#endif
	else{
		re += v;
		return *this;
	}
}

template<typename T>
inline complex_64& complex_64::operator-=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re -= v.re;
		im -= v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re -= v.re;
		im -= v.im;
		return *this;
	
	}
#endif
	else{
		re -= v;
		return *this;
	}
}

template<typename T>
inline complex_64& complex_64::operator/=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re /= v.re;
		im /= v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re /= v.re;
		im /= v.im;
		return *this;
	
	}
#endif
	else{
		re /= v;
		return *this;
	}
}


template complex_64& complex_64::operator*=(complex_64 v);
template complex_64& complex_64::operator*=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator*=(complex_32 v);
template complex_64& complex_64::operator*=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator*=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator*=(int128_t v);
template complex_64& complex_64::operator*=(uint128_t v);
#endif
template complex_64& complex_64::operator*=(double v);
template complex_64& complex_64::operator*=(float v);
template complex_64& complex_64::operator*=(int64_t v);
template complex_64& complex_64::operator*=(uint32_t v);
template complex_64& complex_64::operator*=(int32_t v);
template complex_64& complex_64::operator*=(uint16_t v);
template complex_64& complex_64::operator*=(int16_t v);
template complex_64& complex_64::operator*=(uint8_t v);
template complex_64& complex_64::operator*=(int8_t v);

template complex_64& complex_64::operator+=(complex_64 v);
template complex_64& complex_64::operator+=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator+=(complex_32 v);
template complex_64& complex_64::operator+=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator+=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator+=(int128_t v);
template complex_64& complex_64::operator+=(uint128_t v);
#endif
template complex_64& complex_64::operator+=(double v);
template complex_64& complex_64::operator+=(float v);
template complex_64& complex_64::operator+=(int64_t v);
template complex_64& complex_64::operator+=(uint32_t v);
template complex_64& complex_64::operator+=(int32_t v);
template complex_64& complex_64::operator+=(uint16_t v);
template complex_64& complex_64::operator+=(int16_t v);
template complex_64& complex_64::operator+=(uint8_t v);
template complex_64& complex_64::operator+=(int8_t v);

template complex_64& complex_64::operator-=(complex_64 v);
template complex_64& complex_64::operator-=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator-=(complex_32 v);
template complex_64& complex_64::operator-=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator-=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator-=(int128_t v);
template complex_64& complex_64::operator-=(uint128_t v);
#endif
template complex_64& complex_64::operator-=(double v);
template complex_64& complex_64::operator-=(float v);
template complex_64& complex_64::operator-=(int64_t v);
template complex_64& complex_64::operator-=(uint32_t v);
template complex_64& complex_64::operator-=(int32_t v);
template complex_64& complex_64::operator-=(uint16_t v);
template complex_64& complex_64::operator-=(int16_t v);
template complex_64& complex_64::operator-=(uint8_t v);
template complex_64& complex_64::operator-=(int8_t v);

template complex_64& complex_64::operator/=(complex_64 v);
template complex_64& complex_64::operator/=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator/=(complex_32 v);
template complex_64& complex_64::operator/=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator/=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator/=(int128_t v);
template complex_64& complex_64::operator/=(uint128_t v);
#endif
template complex_64& complex_64::operator/=(double v);
template complex_64& complex_64::operator/=(float v);
template complex_64& complex_64::operator/=(int64_t v);
template complex_64& complex_64::operator/=(uint32_t v);
template complex_64& complex_64::operator/=(int32_t v);
template complex_64& complex_64::operator/=(uint16_t v);
template complex_64& complex_64::operator/=(int16_t v);
template complex_64& complex_64::operator/=(uint8_t v);
template complex_64& complex_64::operator/=(int8_t v);

template<typename T>
inline complex_64 complex_64::operator*(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re * a.re, im * a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re * a.re, im * a.im);
	}
#endif
	else{
		return complex_64(re * a, im);
	}
}

template<typename T>
inline complex_64 complex_64::operator+(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re + a.re, im + a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re + a.re, im + a.im);
	}
#endif
	else{
		return complex_64(re + a, im);
	}
}

template<typename T>
inline complex_64 complex_64::operator-(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re - a.re, im - a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re - a.re, im - a.im);
	}
#endif
	else{
		return complex_64(re - a, im);
	}
}
template<typename T>
inline complex_64 complex_64::operator/(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re / a.re, im / a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re / a.re, im / a.im);
	}
#endif
	else{
		return complex_64(re / a, im);
	}
}


template complex_64 complex_64::operator*(const complex_64 &a);
template complex_64 complex_64::operator*(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator*(const complex_32 &a);
template complex_64 complex_64::operator*(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator*(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator*(const int128_t &a);
template complex_64 complex_64::operator*(const uint128_t &a);
#endif
template complex_64 complex_64::operator*(const double &a);
template complex_64 complex_64::operator*(const float &a);
template complex_64 complex_64::operator*(const int64_t &a);
template complex_64 complex_64::operator*(const uint32_t &a);
template complex_64 complex_64::operator*(const int32_t &a);
template complex_64 complex_64::operator*(const uint16_t &a);
template complex_64 complex_64::operator*(const int16_t &a);
template complex_64 complex_64::operator*(const uint8_t &a);
template complex_64 complex_64::operator*(const int8_t &a);

template complex_64 complex_64::operator+(const complex_64 &a);
template complex_64 complex_64::operator+(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator+(const complex_32 &a);
template complex_64 complex_64::operator+(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator+(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator+(const int128_t &a);
template complex_64 complex_64::operator+(const uint128_t &a);
#endif
template complex_64 complex_64::operator+(const double &a);
template complex_64 complex_64::operator+(const float &a);
template complex_64 complex_64::operator+(const int64_t &a);
template complex_64 complex_64::operator+(const uint32_t &a);
template complex_64 complex_64::operator+(const int32_t &a);
template complex_64 complex_64::operator+(const uint16_t &a);
template complex_64 complex_64::operator+(const int16_t &a);
template complex_64 complex_64::operator+(const uint8_t &a);
template complex_64 complex_64::operator+(const int8_t &a);

template complex_64 complex_64::operator-(const complex_64 &a);
template complex_64 complex_64::operator-(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator-(const complex_32 &a);
template complex_64 complex_64::operator-(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator-(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator-(const int128_t &a);
template complex_64 complex_64::operator-(const uint128_t &a);
#endif
template complex_64 complex_64::operator-(const double &a);
template complex_64 complex_64::operator-(const float &a);
template complex_64 complex_64::operator-(const int64_t &a);
template complex_64 complex_64::operator-(const uint32_t &a);
template complex_64 complex_64::operator-(const int32_t &a);
template complex_64 complex_64::operator-(const uint16_t &a);
template complex_64 complex_64::operator-(const int16_t &a);
template complex_64 complex_64::operator-(const uint8_t &a);
template complex_64 complex_64::operator-(const int8_t &a);


template complex_64 complex_64::operator/(const complex_64 &a);
template complex_64 complex_64::operator/(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator/(const complex_32 &a);
template complex_64 complex_64::operator/(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator/(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator/(const int128_t &a);
template complex_64 complex_64::operator/(const uint128_t &a);
#endif
template complex_64 complex_64::operator/(const double &a);
template complex_64 complex_64::operator/(const float &a);
template complex_64 complex_64::operator/(const int64_t &a);
template complex_64 complex_64::operator/(const uint32_t &a);
template complex_64 complex_64::operator/(const int32_t &a);
template complex_64 complex_64::operator/(const uint16_t &a);
template complex_64 complex_64::operator/(const int16_t &a);
template complex_64 complex_64::operator/(const uint8_t &a);
template complex_64 complex_64::operator/(const int8_t &a);

template<typename T>
complex_64 complex_64::operator*(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re * v.re, im * v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re * v.re, im * v.im);
	}
#endif
	else{
		return complex_64(re * v, im);
	}
}
template<typename T>
complex_64 complex_64::operator+(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re + v.re, im + v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re + v.re, im + v.im);
	}
#endif
	else{
		return complex_64(re + v, im);
	}

}
template<typename T>
complex_64 complex_64::operator-(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re - v.re, im - v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re - v.re, im - v.im);
	}
#endif
	else{
		return complex_64(re - v, im);
	}

}
template<typename T>
complex_64 complex_64::operator/(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_64(re / v.re, im / v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_64(re / v.re, im / v.im);
	}
#endif
	else{
		return complex_64(re / v, im);
	}
}


template complex_64 complex_64::operator*(complex_64 v);
template complex_64 complex_64::operator*(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator*(complex_32 v);
template complex_64 complex_64::operator*(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator*(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator*(int128_t v);
template complex_64 complex_64::operator*(uint128_t v);
#endif
template complex_64 complex_64::operator*(double v);
template complex_64 complex_64::operator*(float v);
template complex_64 complex_64::operator*(int64_t v);
template complex_64 complex_64::operator*(uint32_t v);
template complex_64 complex_64::operator*(int32_t v);
template complex_64 complex_64::operator*(uint16_t v);
template complex_64 complex_64::operator*(int16_t v);
template complex_64 complex_64::operator*(uint8_t v);
template complex_64 complex_64::operator*(int8_t v);

template complex_64 complex_64::operator+(complex_64 v);
template complex_64 complex_64::operator+(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator+(complex_32 v);
template complex_64 complex_64::operator+(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator+(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator+(int128_t v);
template complex_64 complex_64::operator+(uint128_t v);
#endif
template complex_64 complex_64::operator+(double v);
template complex_64 complex_64::operator+(float v);
template complex_64 complex_64::operator+(int64_t v);
template complex_64 complex_64::operator+(uint32_t v);
template complex_64 complex_64::operator+(int32_t v);
template complex_64 complex_64::operator+(uint16_t v);
template complex_64 complex_64::operator+(int16_t v);
template complex_64 complex_64::operator+(uint8_t v);
template complex_64 complex_64::operator+(int8_t v);

template complex_64 complex_64::operator-(complex_64 v);
template complex_64 complex_64::operator-(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator-(complex_32 v);
template complex_64 complex_64::operator-(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator-(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator-(int128_t v);
template complex_64 complex_64::operator-(uint128_t v);
#endif
template complex_64 complex_64::operator-(double v);
template complex_64 complex_64::operator-(float v);
template complex_64 complex_64::operator-(int64_t v);
template complex_64 complex_64::operator-(uint32_t v);
template complex_64 complex_64::operator-(int32_t v);
template complex_64 complex_64::operator-(uint16_t v);
template complex_64 complex_64::operator-(int16_t v);
template complex_64 complex_64::operator-(uint8_t v);
template complex_64 complex_64::operator-(int8_t v);


template complex_64 complex_64::operator/(complex_64 v);
template complex_64 complex_64::operator/(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64 complex_64::operator/(complex_32 v);
template complex_64 complex_64::operator/(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64 complex_64::operator/(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_64 complex_64::operator/(int128_t v);
template complex_64 complex_64::operator/(uint128_t v);
#endif
template complex_64 complex_64::operator/(double v);
template complex_64 complex_64::operator/(float v);
template complex_64 complex_64::operator/(int64_t v);
template complex_64 complex_64::operator/(uint32_t v);
template complex_64 complex_64::operator/(int32_t v);
template complex_64 complex_64::operator/(uint16_t v);
template complex_64 complex_64::operator/(int16_t v);
template complex_64 complex_64::operator/(uint8_t v);
template complex_64 complex_64::operator/(int8_t v);

inline float& complex_64::real(){return re;}
inline const float& complex_64::real() const{return re;}
inline float& complex_64::imag(){return im;}
inline const float& complex_64::imag() const{return im;}

inline complex_64::operator std::complex<float>() const{return std::complex<float>(re, im);}
#ifdef _HALF_FLOAT_SUPPORT_
inline complex_64::operator complex_32() const {return complex_32(static_cast<float16_t>(re), static_cast<float16_t>(im));}
#endif
inline complex_64::operator complex_128() const {return complex_128(static_cast<double>(re), static_cast<double>(im));}


inline bool complex_64::operator==(const complex_64& a) const{
	return a.re == re && a.im == im;
}

template<typename T>
inline complex_64& complex_64::operator=(const T& a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re = static_cast<float>(a.re);
		im = static_cast<float>(a.im);
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re = static_cast<float>(a.re);
		im = static_cast<float>(a.im);
		return *this;
	}
	else if constexpr(std::is_same_v<std::complex<float16_t>>){
		re = static_cast<float>(a.real());
		im = static_cast<float>(a.imag());
		return *this;
	}
#endif
	else if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>){
		re = static_cast<float>(a.real());
		im = static_cast<float>(a.imag());
		return *this;
	} 
	else{
		re = static_cast<float>(a);
		return *this;
	}
}


template complex_64& complex_64::operator=(const complex_64 &a);
template complex_64& complex_64::operator=(const complex_128 &a);
template complex_64& complex_64::operator=(const std::complex<double>&a);
template complex_64& complex_64::operator=(const std::complex<float> &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_64& complex_64::operator=(const float16_t &a);
template complex_64& complex_64::operator=(const complex_32 &a);
template complex_64& complex_64::operator=(const std::complex<float16_t> &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_64& complex_64::operator=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_64& complex_64::operator=(const int128_t &a);
template complex_64& complex_64::operator=(const uint128_t &a);
#endif
template complex_64& complex_64::operator=(const double &a);
template complex_64& complex_64::operator=(const float &a);
template complex_64& complex_64::operator=(const int64_t &a);
template complex_64& complex_64::operator=(const uint32_t &a);
template complex_64& complex_64::operator=(const int32_t &a);
template complex_64& complex_64::operator=(const uint16_t &a);
template complex_64& complex_64::operator=(const int16_t &a);
template complex_64& complex_64::operator=(const uint8_t &a);
template complex_64& complex_64::operator=(const int8_t &a);



complex_128::complex_128()
	:re(0), im(0)
{}

complex_128::complex_128(const double re, const double im)
	:re(re), im(im)
{}

complex_128::complex_128(const std::complex<double> a)
	:re(a.real()), im(a.imag())
{}


inline complex_128& complex_128::operator++(){
	++re;
	++im;
	return *this;
}

inline complex_128 complex_128::operator++(int){
	complex_128 tmp = *this;
	++(*this);
	return tmp;
}

template<typename T>
inline complex_128& complex_128::operator*=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re *= a.re;
		im *= a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re *= a.re;
		im *= a.im;
		return *this;
	}
#endif
	else{
		re *= a;
		return *this;
	}
}


template<typename T>
inline complex_128& complex_128::operator+=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re += a.re;
		im += a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re += a.re;
		im += a.im;
		return *this;
	}
#endif
	else{
		re += a;
		return *this;
	}
}

template<typename T>
inline complex_128& complex_128::operator-=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re -= a.re;
		im -= a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re -= a.re;
		im -= a.im;
		return *this;
	}
#endif	
	else{
		re -= a;
		return *this;
	}
}

template<typename T>
inline complex_128& complex_128::operator/=(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re /= a.re;
		im /= a.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re /= a.re;
		im /= a.im;
		return *this;
	}
#endif
	else{
		re /= a;
		return *this;
	}
}

template complex_128& complex_128::operator*=(const complex_64 &a);
template complex_128& complex_128::operator*=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator*=(const complex_32 &a);
template complex_128& complex_128::operator*=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator*=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator*=(const int128_t &a);
template complex_128& complex_128::operator*=(const uint128_t &a);
#endif
template complex_128& complex_128::operator*=(const double &a);
template complex_128& complex_128::operator*=(const float &a);
template complex_128& complex_128::operator*=(const int64_t &a);
template complex_128& complex_128::operator*=(const uint32_t &a);
template complex_128& complex_128::operator*=(const int32_t &a);
template complex_128& complex_128::operator*=(const uint16_t &a);
template complex_128& complex_128::operator*=(const int16_t &a);
template complex_128& complex_128::operator*=(const uint8_t &a);
template complex_128& complex_128::operator*=(const int8_t &a);

template complex_128& complex_128::operator+=(const complex_64 &a);
template complex_128& complex_128::operator+=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator+=(const complex_32 &a);
template complex_128& complex_128::operator+=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator+=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator+=(const int128_t &a);
template complex_128& complex_128::operator+=(const uint128_t &a);
#endif
template complex_128& complex_128::operator+=(const double &a);
template complex_128& complex_128::operator+=(const float &a);
template complex_128& complex_128::operator+=(const int64_t &a);
template complex_128& complex_128::operator+=(const uint32_t &a);
template complex_128& complex_128::operator+=(const int32_t &a);
template complex_128& complex_128::operator+=(const uint16_t &a);
template complex_128& complex_128::operator+=(const int16_t &a);
template complex_128& complex_128::operator+=(const uint8_t &a);
template complex_128& complex_128::operator+=(const int8_t &a);

template complex_128& complex_128::operator-=(const complex_64 &a);
template complex_128& complex_128::operator-=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator-=(const complex_32 &a);
template complex_128& complex_128::operator-=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator-=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator-=(const int128_t &a);
template complex_128& complex_128::operator-=(const uint128_t &a);
#endif
template complex_128& complex_128::operator-=(const double &a);
template complex_128& complex_128::operator-=(const float &a);
template complex_128& complex_128::operator-=(const int64_t &a);
template complex_128& complex_128::operator-=(const uint32_t &a);
template complex_128& complex_128::operator-=(const int32_t &a);
template complex_128& complex_128::operator-=(const uint16_t &a);
template complex_128& complex_128::operator-=(const int16_t &a);
template complex_128& complex_128::operator-=(const uint8_t &a);
template complex_128& complex_128::operator-=(const int8_t &a);

template complex_128& complex_128::operator/=(const complex_64 &a);
template complex_128& complex_128::operator/=(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator/=(const complex_32 &a);
template complex_128& complex_128::operator/=(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator/=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator/=(const int128_t &a);
template complex_128& complex_128::operator/=(const uint128_t &a);
#endif
template complex_128& complex_128::operator/=(const double &a);
template complex_128& complex_128::operator/=(const float &a);
template complex_128& complex_128::operator/=(const int64_t &a);
template complex_128& complex_128::operator/=(const uint32_t &a);
template complex_128& complex_128::operator/=(const int32_t &a);
template complex_128& complex_128::operator/=(const uint16_t &a);
template complex_128& complex_128::operator/=(const int16_t &a);
template complex_128& complex_128::operator/=(const uint8_t &a);
template complex_128& complex_128::operator/=(const int8_t &a);

template<typename T>
inline complex_128& complex_128::operator*=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re *= v.re;
		im *= v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re *= v.re;
		im *= v.im;
		return *this;
	
	}
#endif
	else{
		re *= v;
		return *this;
	}}


template<typename T>
inline complex_128& complex_128::operator+=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re += v.re;
		im += v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re += v.re;
		im += v.im;
		return *this;
	
	}
#endif
	else{
		re += v;
		return *this;
	}
}

template<typename T>
inline complex_128& complex_128::operator-=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re -= v.re;
		im -= v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re -= v.re;
		im -= v.im;
		return *this;
	
	}
#endif
	else{
		re -= v;
		return *this;
	}
}

template<typename T>
inline complex_128& complex_128::operator/=(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re /= v.re;
		im /= v.im;
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re /= v.re;
		im /= v.im;
		return *this;
	
	}
#endif
	else{
		re /= v;
		return *this;
	}
}


template complex_128& complex_128::operator*=(complex_64 v);
template complex_128& complex_128::operator*=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator*=(complex_32 v);
template complex_128& complex_128::operator*=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator*=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator*=(int128_t v);
template complex_128& complex_128::operator*=(uint128_t v);
#endif
template complex_128& complex_128::operator*=(double v);
template complex_128& complex_128::operator*=(float v);
template complex_128& complex_128::operator*=(int64_t v);
template complex_128& complex_128::operator*=(uint32_t v);
template complex_128& complex_128::operator*=(int32_t v);
template complex_128& complex_128::operator*=(uint16_t v);
template complex_128& complex_128::operator*=(int16_t v);
template complex_128& complex_128::operator*=(uint8_t v);
template complex_128& complex_128::operator*=(int8_t v);

template complex_128& complex_128::operator+=(complex_64 v);
template complex_128& complex_128::operator+=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator+=(complex_32 v);
template complex_128& complex_128::operator+=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator+=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator+=(int128_t v);
template complex_128& complex_128::operator+=(uint128_t v);
#endif
template complex_128& complex_128::operator+=(double v);
template complex_128& complex_128::operator+=(float v);
template complex_128& complex_128::operator+=(int64_t v);
template complex_128& complex_128::operator+=(uint32_t v);
template complex_128& complex_128::operator+=(int32_t v);
template complex_128& complex_128::operator+=(uint16_t v);
template complex_128& complex_128::operator+=(int16_t v);
template complex_128& complex_128::operator+=(uint8_t v);
template complex_128& complex_128::operator+=(int8_t v);

template complex_128& complex_128::operator-=(complex_64 v);
template complex_128& complex_128::operator-=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator-=(complex_32 v);
template complex_128& complex_128::operator-=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator-=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator-=(int128_t v);
template complex_128& complex_128::operator-=(uint128_t v);
#endif
template complex_128& complex_128::operator-=(double v);
template complex_128& complex_128::operator-=(float v);
template complex_128& complex_128::operator-=(int64_t v);
template complex_128& complex_128::operator-=(uint32_t v);
template complex_128& complex_128::operator-=(int32_t v);
template complex_128& complex_128::operator-=(uint16_t v);
template complex_128& complex_128::operator-=(int16_t v);
template complex_128& complex_128::operator-=(uint8_t v);
template complex_128& complex_128::operator-=(int8_t v);

template complex_128& complex_128::operator/=(complex_64 v);
template complex_128& complex_128::operator/=(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator/=(complex_32 v);
template complex_128& complex_128::operator/=(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator/=(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator/=(int128_t v);
template complex_128& complex_128::operator/=(uint128_t v);
#endif
template complex_128& complex_128::operator/=(double v);
template complex_128& complex_128::operator/=(float v);
template complex_128& complex_128::operator/=(int64_t v);
template complex_128& complex_128::operator/=(uint32_t v);
template complex_128& complex_128::operator/=(int32_t v);
template complex_128& complex_128::operator/=(uint16_t v);
template complex_128& complex_128::operator/=(int16_t v);
template complex_128& complex_128::operator/=(uint8_t v);
template complex_128& complex_128::operator/=(int8_t v);

template<typename T>
inline complex_128 complex_128::operator*(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re * a.re, im * a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re * a.re, im * a.im);
		
	}
#endif
	else{
		return complex_128(re * a, im);
	}
}

template<typename T>
inline complex_128 complex_128::operator+(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re + a.re, im + a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re + a.re, im + a.im);
		
	}
#endif
	else{
		return complex_128(re + a, im);
	}
}

template<typename T>
inline complex_128 complex_128::operator-(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re - a.re, im - a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re - a.re, im - a.im);
	}
#endif
	else{
		return complex_128(re - a, im);
	}
}
template<typename T>
inline complex_128 complex_128::operator/(const T &a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re / a.re, im / a.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re / a.re, im / a.im);
	}
#endif
	else{
		return complex_128(re / a, im);
	}
}


template complex_128 complex_128::operator*(const complex_64 &a);
template complex_128 complex_128::operator*(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator*(const complex_32 &a);
template complex_128 complex_128::operator*(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator*(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator*(const int128_t &a);
template complex_128 complex_128::operator*(const uint128_t &a);
#endif
template complex_128 complex_128::operator*(const double &a);
template complex_128 complex_128::operator*(const float &a);
template complex_128 complex_128::operator*(const int64_t &a);
template complex_128 complex_128::operator*(const uint32_t &a);
template complex_128 complex_128::operator*(const int32_t &a);
template complex_128 complex_128::operator*(const uint16_t &a);
template complex_128 complex_128::operator*(const int16_t &a);
template complex_128 complex_128::operator*(const uint8_t &a);
template complex_128 complex_128::operator*(const int8_t &a);

template complex_128 complex_128::operator+(const complex_64 &a);
template complex_128 complex_128::operator+(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator+(const complex_32 &a);
template complex_128 complex_128::operator+(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator+(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator+(const int128_t &a);
template complex_128 complex_128::operator+(const uint128_t &a);
#endif
template complex_128 complex_128::operator+(const double &a);
template complex_128 complex_128::operator+(const float &a);
template complex_128 complex_128::operator+(const int64_t &a);
template complex_128 complex_128::operator+(const uint32_t &a);
template complex_128 complex_128::operator+(const int32_t &a);
template complex_128 complex_128::operator+(const uint16_t &a);
template complex_128 complex_128::operator+(const int16_t &a);
template complex_128 complex_128::operator+(const uint8_t &a);
template complex_128 complex_128::operator+(const int8_t &a);

template complex_128 complex_128::operator-(const complex_64 &a);
template complex_128 complex_128::operator-(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator-(const complex_32 &a);
template complex_128 complex_128::operator-(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator-(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator-(const int128_t &a);
template complex_128 complex_128::operator-(const uint128_t &a);
#endif
template complex_128 complex_128::operator-(const double &a);
template complex_128 complex_128::operator-(const float &a);
template complex_128 complex_128::operator-(const int64_t &a);
template complex_128 complex_128::operator-(const uint32_t &a);
template complex_128 complex_128::operator-(const int32_t &a);
template complex_128 complex_128::operator-(const uint16_t &a);
template complex_128 complex_128::operator-(const int16_t &a);
template complex_128 complex_128::operator-(const uint8_t &a);
template complex_128 complex_128::operator-(const int8_t &a);


template complex_128 complex_128::operator/(const complex_64 &a);
template complex_128 complex_128::operator/(const complex_128 &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator/(const complex_32 &a);
template complex_128 complex_128::operator/(const float16_t &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator/(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator/(const int128_t &a);
template complex_128 complex_128::operator/(const uint128_t &a);
#endif
template complex_128 complex_128::operator/(const double &a);
template complex_128 complex_128::operator/(const float &a);
template complex_128 complex_128::operator/(const int64_t &a);
template complex_128 complex_128::operator/(const uint32_t &a);
template complex_128 complex_128::operator/(const int32_t &a);
template complex_128 complex_128::operator/(const uint16_t &a);
template complex_128 complex_128::operator/(const int16_t &a);
template complex_128 complex_128::operator/(const uint8_t &a);
template complex_128 complex_128::operator/(const int8_t &a);

template<typename T>
complex_128 complex_128::operator*(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re * v.re, im * v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re * v.re, im * v.im);
	}
#endif
	else{
		return complex_128(re * v, im);
	}
}
template<typename T>
complex_128 complex_128::operator+(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re + v.re, im + v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re + v.re, im + v.im);
	}
#endif
	else{
		return complex_128(re + v, im);
	}

}
template<typename T>
complex_128 complex_128::operator-(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re - v.re, im - v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re - v.re, im - v.im);
	}
#endif
	else{
		return complex_128(re - v, im);
	}

}
template<typename T>
complex_128 complex_128::operator/(T v){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		return complex_128(re / v.re, im / v.im);
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		return complex_128(re / v.re, im / v.im);
	}
#endif
	else{
		return complex_128(re / v, im);
	}
}


template complex_128 complex_128::operator*(complex_64 v);
template complex_128 complex_128::operator*(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator*(complex_32 v);
template complex_128 complex_128::operator*(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator*(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator*(int128_t v);
template complex_128 complex_128::operator*(uint128_t v);
#endif
template complex_128 complex_128::operator*(double v);
template complex_128 complex_128::operator*(float v);
template complex_128 complex_128::operator*(int64_t v);
template complex_128 complex_128::operator*(uint32_t v);
template complex_128 complex_128::operator*(int32_t v);
template complex_128 complex_128::operator*(uint16_t v);
template complex_128 complex_128::operator*(int16_t v);
template complex_128 complex_128::operator*(uint8_t v);
template complex_128 complex_128::operator*(int8_t v);

template complex_128 complex_128::operator+(complex_64 v);
template complex_128 complex_128::operator+(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator+(complex_32 v);
template complex_128 complex_128::operator+(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator+(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator+(int128_t v);
template complex_128 complex_128::operator+(uint128_t v);
#endif
template complex_128 complex_128::operator+(double v);
template complex_128 complex_128::operator+(float v);
template complex_128 complex_128::operator+(int64_t v);
template complex_128 complex_128::operator+(uint32_t v);
template complex_128 complex_128::operator+(int32_t v);
template complex_128 complex_128::operator+(uint16_t v);
template complex_128 complex_128::operator+(int16_t v);
template complex_128 complex_128::operator+(uint8_t v);
template complex_128 complex_128::operator+(int8_t v);

template complex_128 complex_128::operator-(complex_64 v);
template complex_128 complex_128::operator-(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator-(complex_32 v);
template complex_128 complex_128::operator-(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator-(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator-(int128_t v);
template complex_128 complex_128::operator-(uint128_t v);
#endif
template complex_128 complex_128::operator-(double v);
template complex_128 complex_128::operator-(float v);
template complex_128 complex_128::operator-(int64_t v);
template complex_128 complex_128::operator-(uint32_t v);
template complex_128 complex_128::operator-(int32_t v);
template complex_128 complex_128::operator-(uint16_t v);
template complex_128 complex_128::operator-(int16_t v);
template complex_128 complex_128::operator-(uint8_t v);
template complex_128 complex_128::operator-(int8_t v);


template complex_128 complex_128::operator/(complex_64 v);
template complex_128 complex_128::operator/(complex_128 v);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128 complex_128::operator/(complex_32 v);
template complex_128 complex_128::operator/(float16_t v);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128 complex_128::operator/(float128_t v);
#endif
#ifdef __SIZEOF_INT128__
template complex_128 complex_128::operator/(int128_t v);
template complex_128 complex_128::operator/(uint128_t v);
#endif
template complex_128 complex_128::operator/(double v);
template complex_128 complex_128::operator/(float v);
template complex_128 complex_128::operator/(int64_t v);
template complex_128 complex_128::operator/(uint32_t v);
template complex_128 complex_128::operator/(int32_t v);
template complex_128 complex_128::operator/(uint16_t v);
template complex_128 complex_128::operator/(int16_t v);
template complex_128 complex_128::operator/(uint8_t v);
template complex_128 complex_128::operator/(int8_t v);


inline double& complex_128::real(){return re;}
inline const double& complex_128::real() const{return re;}
inline double& complex_128::imag(){return im;}
inline const double& complex_128::imag() const{return im;}

inline complex_128::operator std::complex<double>() const{return std::complex<double>(re, im);}
#ifdef _HALF_FLOAT_SUPPORT_
inline complex_128::operator complex_32() const {return complex_32(static_cast<float16_t>(re), static_cast<float16_t>(im));}
#endif
inline complex_128::operator complex_64() const {return complex_64(static_cast<float>(re), static_cast<float>(im));}



inline bool complex_128::operator==(const complex_128& a) const{
	return a.re == re && a.im == im;
}

template<typename T>
inline complex_128& complex_128::operator=(const T& a){
	if constexpr(std::is_same_v<T, complex_64> || std::is_same_v<T, complex_128>){
		re = static_cast<double>(a.re);
		im = static_cast<double>(a.im);
		return *this;
	}
#ifdef _HALF_FLOAT_SUPPORT_
	else if constexpr(std::is_same_v<T, complex_32>){
		re = static_cast<double>(a.re);
		im = static_cast<double>(a.im);
		return *this;
	}
	else if constexpr(std::is_same_v<std::complex<float16_t>>){
		re = static_cast<double>(a.real());
		im = static_cast<double>(a.imag());
		return *this;
	}
#endif
	else if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>){
		re = static_cast<double>(a.real());
		im = static_cast<double>(a.imag());
		return *this;
	} 
	else{
		re = static_cast<double>(a);
		return *this;
	}
}


template complex_128& complex_128::operator=(const complex_64 &a);
template complex_128& complex_128::operator=(const complex_128 &a);
template complex_128& complex_128::operator=(const std::complex<double>&a);
template complex_128& complex_128::operator=(const std::complex<float> &a);
#ifdef _HALF_FLOAT_SUPPORT_
template complex_128& complex_128::operator=(const float16_t &a);
template complex_128& complex_128::operator=(const complex_32 &a);
template complex_128& complex_128::operator=(const std::complex<float16_t> &a);
#endif
#ifdef _128_FLOAT_SUPPORT_
template complex_128& complex_128::operator=(const float128_t &a);
#endif
#ifdef __SIZEOF_INT128__
template complex_128& complex_128::operator=(const int128_t &a);
template complex_128& complex_128::operator=(const uint128_t &a);
#endif
template complex_128& complex_128::operator=(const double &a);
template complex_128& complex_128::operator=(const float &a);
template complex_128& complex_128::operator=(const int64_t &a);
template complex_128& complex_128::operator=(const uint32_t &a);
template complex_128& complex_128::operator=(const int32_t &a);
template complex_128& complex_128::operator=(const uint16_t &a);
template complex_128& complex_128::operator=(const int16_t &a);
template complex_128& complex_128::operator=(const uint8_t &a);
template complex_128& complex_128::operator=(const int8_t &a);




}
