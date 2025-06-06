#include "Types.h"
#include <ostream>

#include <type_traits>
#include <valarray>
#include "../convert/std_convert.h"

namespace nt{
//already defined
/* #ifdef _HALF_FLOAT_SUPPORT_ */

/* std::ostream& operator<<(std::ostream& os, const float16_t& val){ */
/* 	os << convert::convert<float>(val); */
/* 	return os; */
/* } */

/* #endif */

#ifdef _128_FLOAT_SUPPORT_
std::ostream& operator<<(std::ostream& os, const float128_t& val){
	os << convert::convert<double>(val);
	return os;
}

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

bool operator==(const uint_bool_t& a, const uint_bool_t& b){return a.value == b.value;}
bool operator==(const uint_bool_t& a, const bool& b){return (a.value == 1 && b) || (a.value == 0 && !b);}
bool operator==(const bool& a, const uint_bool_t& b){return (b.value == 1 && a) || (b.value == 0 && !a);}


template<typename T>
my_complex<T>::my_complex()
	:re(0), im(0)
{}

template<typename T>
my_complex<T>::my_complex(const T& re, const T& im)
	:re(re), im(im)
{}

template<typename T>
my_complex<T>::my_complex(const std::complex<T>& c)
	:re(c.real()), im(c.imag())
{}

template<typename T>
my_complex<T>& my_complex<T>::operator+=(T val){
	re += val;
	return *this;
}

template<typename T>
my_complex<T>& my_complex<T>::operator*=(T val){
	re *= val;
	return *this;
}

template<typename T>
my_complex<T>& my_complex<T>::operator-=(T val){
	re -= val;
	return *this;
}

template<typename T>
my_complex<T>& my_complex<T>::operator/=(T val){
	re /= val;
	return *this;
}


template <typename T>
my_complex<T>& my_complex<T>::operator+=(const my_complex<float>& val) {
    re += convert::convert<T>(val.real());
    im += convert::convert<T>(val.imag());
    return *this;
}

template <typename T>
my_complex<T>& my_complex<T>::operator+=(const my_complex<double>& val) {
    re += convert::convert<T>(val.real());
    im += convert::convert<T>(val.imag());
    return *this;
}

#ifdef _HALF_FLOAT_SUPPORT_
template <typename T>
my_complex<T>& my_complex<T>::operator+=(const my_complex<float16_t>& val) {
    re += convert::convert<T>(val.real());
    im += convert::convert<T>(val.imag());
    return *this;
}
#endif
template <typename T>
my_complex<T>& my_complex<T>::operator*=(const my_complex<float>& val) {
    re *= convert::convert<T>(val.real());
    im *= convert::convert<T>(val.imag());
    return *this;
}

template <typename T>
my_complex<T>& my_complex<T>::operator*=(const my_complex<double>& val) {
    re *= convert::convert<T>(val.real());
    im *= convert::convert<T>(val.imag());
    return *this;
}

#ifdef _HALF_FLOAT_SUPPORT_
template <typename T>
my_complex<T>& my_complex<T>::operator*=(const my_complex<float16_t>& val) {
    re *= convert::convert<T>(val.real());
    im *= convert::convert<T>(val.imag());
    return *this;
}
#endif


template <typename T>
my_complex<T>& my_complex<T>::operator-=(const my_complex<float>& val) {
    re -= convert::convert<T>(val.real());
    im -= convert::convert<T>(val.imag());
    return *this;
}

template <typename T>
my_complex<T>& my_complex<T>::operator-=(const my_complex<double>& val) {
    re -= convert::convert<T>(val.real());
    im -= convert::convert<T>(val.imag());
    return *this;
}

#ifdef _HALF_FLOAT_SUPPORT_
template <typename T>
my_complex<T>& my_complex<T>::operator-=(const my_complex<float16_t>& val) {
    re -= convert::convert<T>(val.real());
    im -= convert::convert<T>(val.imag());
    return *this;
}
#endif


template <typename T>
my_complex<T>& my_complex<T>::operator/=(const my_complex<float>& val) {
    re /= convert::convert<T>(val.real());
    im /= convert::convert<T>(val.imag());
    return *this;
}

template <typename T>
my_complex<T>& my_complex<T>::operator/=(const my_complex<double>& val) {
    re /= convert::convert<T>(val.real());
    im /= convert::convert<T>(val.imag());
    return *this;
}

#ifdef _HALF_FLOAT_SUPPORT_
template <typename T>
my_complex<T>& my_complex<T>::operator/=(const my_complex<float16_t>& val) {
    re /= convert::convert<T>(val.real());
    im /= convert::convert<T>(val.imag());
    return *this;
}
#endif

template<typename T>
my_complex<T> my_complex<T>::operator+(T val) const{
	return my_complex<T>(re + val, im);
}

template<typename T>
my_complex<T> my_complex<T>::operator*(T val) const{
	return my_complex<T>(re * val, im);
}

template<typename T>
my_complex<T> my_complex<T>::operator-(T val) const{
	return my_complex<T>(re - val, im);
}

template<typename T>
my_complex<T> my_complex<T>::operator/(T val) const{
	return my_complex<T>(re / val, im);
}

#define NT_MAKE_OTHER_COMPLEX_OPERATOR(operation)\
template<typename T>\
my_complex<T> my_complex<T>::operator##operation(const my_complex<float>& val) const {\
    if constexpr (std::is_same_v<T, float16_t>){\
        
    }
}


#define NT_MAKE_OTHER_COMPLEX_OPERATOR(operation)\
template<typename T>\
my_complex<T> my_complex<T>::operator##operation(const my_complex<float>& val) const {\
    if constexpr (std::is_same_v<T, float16_t>){\
        return my_complex<float16_t>(re + _NT_FLOAT32_TO_FLOAT16_(val.real()), im + _NT_FLOAT32_TO_FLOAT16_(val.imag()));\
    }\
    else if constexpr(std::is_same_v<T, float>){\
        return my_complex<float>(re + val.real(), im + val.imag());\
    }\
    else{\
        return my_complex<T>(re + static_cast<T>(val.real()), im + static_cast<T>(val.imag()));\
    }\
}\
\
template<typename T>\
my_complex<T> my_complex<T>::operator##operation(const my_complex<double>& val) const {\
    if constexpr (std::is_same_v<T, float16_t>){\
        return my_complex<float16_t>(re + _NT_FLOAT32_TO_FLOAT16_(static_cast<float>(val.real())), im + _NT_FLOAT32_TO_FLOAT16_(static_cast<float>(val.real())));\
    }\
    else if constexpr(std::is_same_v<T, double>){\
        return my_complex<double>(re + val.real(), im + val.imag());\
    }\
    else{\
        return my_complex<T>(re + static_cast<T>(val.real()), im + static_cast<T>(val.imag()));\
    }\
}\
\
template<typename T>\
my_complex<T> my_complex<T>::operator##operation(const my_complex<float16_t>& val) const {\
    if constexpr (std::is_same_v<T, float16_t>){\
        return my_complex<float16_t>(re + val.real()), im + val.real());\
    }else{\
        return my_complex<T>(re + static_cast<T>(val.real()), im + static_cast<T>(val.imag()));\
    }\
}\

NT_MAKE_OTHER_COMPLEX_OPERATOR(+);
NT_MAKE_OTHER_COMPLEX_OPERATOR(*);
NT_MAKE_OTHER_COMPLEX_OPERATOR(-);
NT_MAKE_OTHER_COMPLEX_OPERATOR(/);


template<typename T>
T& my_complex<T>::real(){return re;}
template<typename T>
const T& my_complex<T>::real() const {return re;}
template<typename T>
T& my_complex<T>::imag(){return im;}
template<typename T>
const T& my_complex<T>::imag() const {return im;}

template<typename T>
my_complex<T>& my_complex<T>::operator++(){
	++re;
	++im;
	return *this;
}

template<typename T>
my_complex<T> my_complex<T>::operator++(int){
	my_complex<T> tmp  = *this;
	++(*this);
	return tmp;
}

template<typename T>
bool my_complex<T>::operator==(const my_complex<T>& c) const {return (re == c.re) && (im == c.im);}

template<typename T>
bool my_complex<T>::operator!=(const my_complex<T>& c) const {return (re != c.re) || (im != c.im);}

template<typename T>
my_complex<T>::operator std::complex<T>() const {return std::complex<T>(re, im);}


template<typename T>
my_complex<T>& my_complex<T>::operator=(const T& val){
	re = val;
	return *this;
}

template<typename T>
my_complex<T>& my_complex<T>::operator=(const my_complex<T>& val){
	re = val.re;
	im = val.im;
	return *this;
}


template<typename T>
my_complex<T> my_complex<T>::inverse() const{
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, float128_t>){
	    return my_complex<T>(T(1.0)/re, T(1.0)/im);
    }
}

template<typename T>
my_complex<T>& my_complex<T>::inverse_(){
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, float128_t>){
	re = T(1.0)/re;
	im = T(1.0)/im;
    }
	return *this;
}


template<typename T>
template<typename X, std::enable_if_t<!std::is_same_v<T, X>, bool>>
my_complex<T>::operator my_complex<X>() const {return my_complex<X>(convert::convert<X>(re), convert::convert<X>(im));}

template my_complex<float>::operator my_complex<double>() const;
template my_complex<double>::operator my_complex<float>() const;
template my_complex<float>::operator my_complex<float16_t>() const;
template my_complex<double>::operator my_complex<float16_t>() const;
template my_complex<float16_t>::operator my_complex<float>() const;
template my_complex<float16_t>::operator my_complex<double>() const;



/* template<> my_complex<float>::operator<double> my_complex<double>() const {return my_complex<double>(convert::convert<double>(re), convert::convert<double>(im));} */
/* my_complex<double>::operator my_complex<float>() const {return my_complex<float>(convert::convert<float>(re), convert::convert<float>(im));} */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* my_complex<float16_t>::operator my_complex<float>() const {return my_complex<float>(convert::convert<float>(re), convert::convert<float>(im));} */
/* my_complex<float16_t>::operator my_complex<double>() const {return my_complex<double>(convert::convert<double>(re), convert::convert<double>(im));} */
/* my_complex<float>::operator my_complex<float16_t>() const {return my_complex<float16_t>(convert::convert<float16_t>(re), convert::convert<float16_t>(im));} */
/* my_complex<double>::operator my_complex<float16_t>() const {return my_complex<float16_t>(convert::convert<float16_t>(re), convert::convert<float16_t>(im));} */
/* #endif */

template<typename T>
template<typename X, std::enable_if_t<!std::is_same_v<T, X>, bool>>
my_complex<T>& my_complex<T>::operator=(const my_complex<X>& c){
	re = convert::convert<T>(c.real());
	im = convert::convert<T>(c.imag());
	return *this;
}

template my_complex<float>& my_complex<float>::operator=(const my_complex<double>& c);
template my_complex<double>& my_complex<double>::operator=(const my_complex<float>& c);
template my_complex<float>& my_complex<float>::operator=(const my_complex<float16_t>& c);
template my_complex<double>& my_complex<double>::operator=(const my_complex<float16_t>& c);
template my_complex<float16_t>& my_complex<float16_t>::operator=(const my_complex<float>& c);
template my_complex<float16_t>& my_complex<float16_t>::operator=(const my_complex<double>& c);

/* template<> my_complex<float>& my_complex<float>::operator=<double>(const my_complex<double>& c){ */
/* 	re = convert::convert<float>(c.real()); */
/* 	im = convert::convert<float>(c.imag()); */
/* 	return *this; */	
/* } */


/* template<> my_complex<double>& my_complex<double>::operator=<float>(const my_complex<float>& c){ */
/* 	re = convert::convert<double>(c.real()); */
/* 	im = convert::convert<double>(c.imag()); */
/* 	return *this; */	
/* } */

/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* template<> my_complex<float16_t>& my_complex<float16_t>::operator=<float>(const my_complex<float>& c){ */
/* 	re = convert::convert<float16_t>(c.real()); */
/* 	im = convert::convert<float16_t>(c.imag()); */
/* 	return *this; */	
/* } */
/* template<> my_complex<float16_t>& my_complex<float16_t>::operator=<double>(const my_complex<double>& c){ */
/* 	re = convert::convert<float16_t>(c.real()); */
/* 	im = convert::convert<float16_t>(c.imag()); */
/* 	return *this; */	
/* } */

/* template<> my_complex<float>& my_complex<float>::operator=<float16_t>(const my_complex<float16_t>& c){ */
/* 	re = convert::convert<float>(c.real()); */
/* 	im = convert::convert<float>(c.imag()); */
/* 	return *this; */	
/* } */

/* template<> my_complex<double>& my_complex<double>::operator=<float16_t(const my_complex<float16_t>& c){ */
/* 	re = convert::convert<double>(c.real()); */
/* 	im = convert::convert<double>(c.imag()); */
/* 	return *this; */	
/* } */


/* #endif */

template<typename T>
bool my_complex<T>::operator<(const my_complex<T>& c) const{
	return (re < c.re) && (im < c.im);
}

template<typename T>
bool my_complex<T>::operator>(const my_complex<T>& c) const{
	return (re > c.re) && (im > c.im);
}

template<typename T>
bool my_complex<T>::operator>=(const my_complex<T>& c) const{
	return (re >= c.re) && (im >= c.im);
}

template<typename T>
bool my_complex<T>::operator<=(const my_complex<T>& c) const{
	return (re <= c.re) && (im <= c.im);
}

template class my_complex<float>;
template class my_complex<double>;
template class my_complex<float16_t>;

}
