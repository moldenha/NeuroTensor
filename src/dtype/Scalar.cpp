#include "Scalar.h"
#include "../types/Types.h"
#include "../convert/Convert.h"
#include "DType_enum.h"

#include <type_traits>

namespace nt{

Scalar::Scalar(const Scalar& s)
	:dtype(s.dtype),
	v(s.v)
	{}

Scalar& Scalar::operator=(const Scalar &s){
	dtype = s.dtype;
	if(isComplex()){
		v.c = s.v.c;	
	}
	else if(isFloatingPoint()){
		v.d = s.v.d;
	}
	else if(isIntegral()){
		v.i = s.v.i;
	}
	else{
		v.i = s.v.i;
	}
	return *this;
}

Scalar::Scalar()
	:dtype(DType::Integer)
{v.i = 0;}

/* template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_integer_v<DTypeFuncs::type_to_dtype<T>>, bool>> */
/* Scalar::Scalar(T vv) */
/* :dtype(DTypeFuncs::type_to_dtype<T>) */
/* {v.i = convert::convert<decltype(v.i)>(vv);} */

/* template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_complex_v<DTypeFuncs::type_to_dtype<T>>, bool>> */
/* Scalar::Scalar(T vv) */
/* :dtype(DTypeFuncs::type_to_dtype<T>) */
/* {v.c = convert::convert<decltype(v.c)>(vv);} */

/* template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_floating_v<DTypeFuncs::type_to_dtype<T>>, bool>> */
/* Scalar::Scalar(T vv) */
/* :dtype(DTypeFuncs::type_to_dtype<T>) */
/* {v.d = convert::convert<decltype(v.d)>(vv);} */

/* template<typename T, std::enable_if_t<std::is_same_v<T, bool>, bool>> */
/* Scalar::Scalar(T vv) */
/* :dtype(DTypeFuncs::type_to_dtype<T>) */
/* {v.i = vv ? 1 : 0;} */

/* template<typename T, std::enable_if_t<std::is_same_v<T, uint_bool_t>, bool>> */
/* Scalar::Scalar(T vv) */
/* :dtype(DTypeFuncs::type_to_dtype<T>) */
/* {v.i = (vv == true) ? 1 : 0;} */


template<>
Scalar::Scalar(int32_t vv)
	:dtype(DType::Integer)
{v.i = convert::convert<decltype(v.i)>(vv);} 

#ifdef __SIZEOF_INT128__
template<>
Scalar::Scalar(int128_t vv)
	:dtype(DType::int128)
{v.i = convert::convert<decltype(v.i)>(vv);} 

template<>
Scalar::Scalar(uint128_t vv)
	:dtype(DType::uint128)
{v.i = convert::convert<decltype(v.i)>(vv);} 
#endif
#ifdef _HALF_FLOAT_SUPPORT_
template <>
Scalar::Scalar(float16_t vv)
	:dtype(DType::Float16)
{v.d = convert::convert<decltype(v.d)>(vv);}

template <>
Scalar::Scalar(complex_32 vv)
	:dtype(DType::Complex32)
{v.c = convert::convert<decltype(v.c)>(vv);}
#endif
#ifdef _128_FLOAT_SUPPORT_
template <>
Scalar::Scalar(float128_t vv)
	:dtype(DType::Float128)
{v.d = convert::convert<decltype(v.d)>(vv);}
#endif
template <>
Scalar::Scalar(double vv)
	:dtype(DType::Double)
{v.d = convert::convert<decltype(v.d)>(vv);}
template <>
Scalar::Scalar(float vv)
	:dtype(DType::Float)
{v.d = convert::convert<decltype(v.d)>(vv);}
template <>
Scalar::Scalar(uint32_t vv)
	:dtype(DType::uint32)
{v.i = convert::convert<decltype(v.i)>(vv);}
template <>
Scalar::Scalar(complex_64 vv)
	:dtype(DType::Complex64)
{v.c = convert::convert<decltype(v.c)>(vv);}
template <>
Scalar::Scalar(complex_128 vv)
	:dtype(DType::Complex128)
{v.c = convert::convert<decltype(v.c)>(vv);}
template <>
Scalar::Scalar(uint8_t vv)
	:dtype(DType::uint8)
{v.i = convert::convert<decltype(v.i)>(vv);}
template <>
Scalar::Scalar(int8_t vv)
	:dtype(DType::int8)
{v.i = convert::convert<decltype(v.i)>(vv);}
template <>
Scalar::Scalar(int16_t vv)
	:dtype(DType::int16)
{v.i = convert::convert<decltype(v.i)>(vv);}
template <>
Scalar::Scalar(uint16_t vv)
	:dtype(DType::uint16)
{v.i = convert::convert<decltype(v.i)>(vv);}
template <>
Scalar::Scalar(int64_t vv)
	:dtype(DType::int64)
{v.i = convert::convert<decltype(v.i)>(vv);} 
template <>
Scalar::Scalar(uint_bool_t vv)
	:dtype(DType::Bool)
{v.i = (vv == true) ? 1 : 0;}
template <>
Scalar::Scalar(bool vv)
	:dtype(DType::Bool)
{v.i = vv ? 1 : 0;}



bool Scalar::isComplex() const{return DTypeFuncs::is_complex(dtype);}
bool Scalar::isFloatingPoint() const{return DTypeFuncs::is_floating(dtype);}
bool Scalar::isIntegral() const{return DTypeFuncs::is_integer(dtype);}
bool Scalar::isBoolean() const{return dtype == DType::Bool;}
bool Scalar::isZero() const {
	if(isComplex()){
		return v.c == complex_64(0,0);
	}
	else if(isFloatingPoint()){
		return v.d == 0;
	}
	else if(isIntegral()){
		return v.i == 0;
	}
	return v.i == 0;
}

DType Scalar::type() const{
	if(isComplex())
		return DType::cdouble;
	else if(isFloatingPoint()){
#ifdef _128_FLOAT_SUPPORT_
		return DType::Float128;
#else
		return DType::Double;
#endif
	}
	else if(isIntegral()){
#ifdef __SIZEOF_INT128__
		return DType::int128;
#else
		return DType::LongLong;
#endif
	}
	else if(isBoolean())
		return DType::Bool;
	else
		throw std::runtime_error("Unknown scalar type");
}



/* Scalar Scalar::operator+(const Scalar& s) const{ */
/* 	if(isZero()){return s;} */
/* 	if(s.isZero()){return *this;} */
/* 	if(isComplex()){ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>() + v.c; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>() + v.d; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>() + v.d; */
/* #endif */

/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>() + v.i; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>() + v.i; */
/* #endif */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */

/* Scalar Scalar::operator-(const Scalar& s) const{ */
/* 	if(isZero()){return s;} */
/* 	if(s.isZero()){return *this;} */
/* 	if(isComplex()){ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>() - v.c; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>() - v.d; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>() - v.d; */
/* #endif */

/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>() - v.i; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>() - v.i; */
/* #endif */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */

/* Scalar Scalar::operator*(const Scalar& s) const{ */
/* 	if(isZero()){return *this;} */
/* 	if(s.isZero()){return s;} */
/* 	if(isComplex()){ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>() * v.c; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>() * v.d; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>() * v.d; */
/* #endif */

/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>() * v.i; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>() * v.i; */
/* #endif */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */


/* Scalar Scalar::operator/(const Scalar& s) const{ */
/* 	if(isZero()){return s;} */
/* 	if(s.isZero()){return *this;} */
/* 	if(isComplex()){ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>() / v.c; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>() / v.d; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>() / v.d; */
/* #endif */

/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>() / v.i; */
/* #else */
/* 		return s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>() / v.i; */
/* #endif */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */


/* Scalar& Scalar::operator+=(const Scalar& s){ */
/* 	if(isZero()){*this = s;} */
/* 	if(s.isZero()){return *this;} */
/* 	if(isComplex()){ */
/* 		v.c += s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>(); */
/* 		return *this; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		v.d +=  s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>(); */ 
/* #else */
/* 		v.d += s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>(); */ 
/* #endif */
/* 		return *this; */
/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		v.i += s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>(); */ 
/* #else */
/* 		v.i += s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>(); */
/* #endif */
/* 		return *this; */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */

/* Scalar& Scalar::operator-=(const Scalar& s){ */
/* 	if(isZero()){*this = s;} */
/* 	if(s.isZero()){return *this;} */
/* 	if(isComplex()){ */
/* 		v.c -= s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>(); */
/* 		return *this; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		v.d -=  s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>(); */ 
/* #else */
/* 		v.d -= s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>(); */ 
/* #endif */
/* 		return *this; */
/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		v.i -= s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>(); */ 
/* #else */
/* 		v.i -= s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>(); */
/* #endif */
/* 		return *this; */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */

/* Scalar& Scalar::operator*=(const Scalar& s){ */
/* 	if(isZero()){return *this;} */
/* 	if(isComplex()){ */
/* 		v.c *= s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>(); */
/* 		return *this; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		v.d *=  s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>(); */ 
/* #else */
/* 		v.d *= s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>(); */ 
/* #endif */
/* 		return *this; */
/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		v.i *= s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>(); */ 
/* #else */
/* 		v.i *= s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>(); */
/* #endif */
/* 		return *this; */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */

/* Scalar& Scalar::operator/=(const Scalar& s){ */
/* 	if(isZero()){return *this;} */
/* 	if(s.isZero()){ */
/* 		throw std::runtime_error("Cannot divide by 0"); */
/* 		return *this; */
/* 	} */
/* 	if(isComplex()){ */
/* 		v.c /= s.to<DTypeFuncs::dtype_to_type_t<DType::cdouble>>(); */
/* 		return *this; */
/* 	} */
/* 	if(isFloatingPoint()){ */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* 		v.d /=  s.to<DTypeFuncs::dtype_to_type_t<DType::Float128>>(); */ 
/* #else */
/* 		v.d /= s.to<DTypeFuncs::dtype_to_type_t<DType::Float64>>(); */ 
/* #endif */
/* 		return *this; */
/* 	} */
/* 	if(isIntegral()){ */
/* #ifdef __SIZEOF_INT128__ */
/* 		v.i /= s.to<DTypeFuncs::dtype_to_type_t<DType::int128>>(); */ 
/* #else */
/* 		v.i /= s.to<DTypeFuncs::dtype_to_type_t<DType::LongLong>>(); */
/* #endif */
/* 		return *this; */

/* 	} */
/* 	throw std::runtime_error("Cannot perform operation on boolean"); */
/* 	return *this; */
/* } */

template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_integer_v<DTypeFuncs::type_to_dtype<T>>, bool>>
T Scalar::to() const{
	if(isComplex())
		return convert::convert<T>(v.c);
	else if(isFloatingPoint())
		return convert::convert<T>(v.d);
	return static_cast<T>(v.i);
}

template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_complex_v<DTypeFuncs::type_to_dtype<T>>, bool>>
T Scalar::to() const{
	if(isComplex())
		return T(v.c.real(), v.c.imag());
	else if(isFloatingPoint())
		return convert::convert<T>(v.d);
	return convert::convert<T>(v.i);
}


template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_floating_v<DTypeFuncs::type_to_dtype<T>>, bool>>
T Scalar::to() const{
	if(isComplex())
		return convert::convert<T>(v.c);
	else if(isFloatingPoint())
		return convert::convert<T>(v.d);
	return convert::convert<T>(v.i);
}

template<typename T, std::enable_if_t<DTypeFuncs::type_to_dtype<T> == DType::Bool, bool>>
uint_bool_t Scalar::to() const{
	if(isComplex())
		return uint_bool_t(v.c.real() > 0);
	else if(isFloatingPoint())
		return uint_bool_t(v.d > 0);
	return uint_bool_t(v.i > 0);
}


template int32_t Scalar::to<int32_t>() const;
template uint32_t Scalar::to<uint32_t>() const;
template uint16_t Scalar::to<uint16_t>() const;
template int16_t Scalar::to<int16_t>() const;
template int8_t Scalar::to<int8_t>() const;
template uint8_t Scalar::to<uint8_t>() const;
template int64_t Scalar::to<int64_t>() const;
#ifdef __SIZEOF_INT128__
template uint128_t Scalar::to<uint128_t>() const;
template int128_t Scalar::to<int128_t>() const;
#endif
template float Scalar::to<float>() const;
template double Scalar::to<double>() const;
#ifdef _HALF_FLOAT_SUPPORT_
template float16_t Scalar::to<float16_t>() const;
template complex_32 Scalar::to<complex_32>() const;
#endif
#ifdef _128_FLOAT_SUPPORT_
template float128_t Scalar::to<float128_t>() const;
#endif
template complex_64 Scalar::to<complex_64>() const;
template complex_128 Scalar::to<complex_128>() const;
template uint_bool_t Scalar::to<uint_bool_t>() const;


Scalar Scalar::inverse() const{
	if(isComplex()){
		return Scalar(v.c.inverse());
	}
	if(isFloatingPoint()){
		return Scalar((1/v.d));
	}
	if(isIntegral()){
		return Scalar(1/(double)(v.i));
	}
	return Scalar(0);
}


ScalarRef& ScalarRef::operator=(const Tensor& val){
	if(dtype != DType::TensorObj)
		return *this;
	get<DType::TensorObj>() = val;
	return *this;
}

std::ostream& operator<<(std::ostream& os, const ScalarRef& s){
	switch(s.dtype){
		case DType::Float:
			return os << s.data.f32.get();
		case DType::Double:
			return os << s.data.f64.get();
		case DType::Integer:
			return os << s.data.i32.get();
		case DType::uint32:
			return os << s.data.i32_u.get();
		case DType::int64:
			return os << s.data.i64.get();
		case DType::int16:
			return os << s.data.i16.get();
		case DType::uint16:
			return os << s.data.i16_u.get();
		case DType::int8:
			return os << s.data.i8.get();
		case DType::uint8:
			return os << s.data.i8_u.get();
		case DType::Complex64:
			return os << s.data.c64.get();
		case DType::Complex128:
			return os << s.data.c128.get();
		case DType::Bool:
			return os << s.data.b.get();
		case DType::TensorObj:
			return os << s.data.t.get();
#ifdef __SIZEOF_INT128__
		case DType::int128:
			return os << s.data.i128.get();
		case DType::uint128:
			return os << s.data.i128_u.get();
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return os << s.data.f16.get();
		case DType::Complex32:
			return os << s.data.c32.get();
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return os << s.data.f128.get();
#endif
	}
	return os;
}

std::ostream& operator<<(std::ostream& os, const Scalar& s){
	if(s.isComplex())
		return os << s.v.c;
	if(s.isFloatingPoint())
		return os << s.v.d;
	if(s.isIntegral())
		return os << s.v.i;
	return os << std::boolalpha << bool(s.v.i) << std::noboolalpha;
}

std::ostream& operator<<(std::ostream& os, const ConstScalarRef& s){
	switch(s.dtype){
		case DType::Float:
			return os << s.data.f32.get();
		case DType::Double:
			return os << s.data.f64.get();
		case DType::Integer:
			return os << s.data.i32.get();
		case DType::uint32:
			return os << s.data.i32_u.get();
		case DType::int64:
			return os << s.data.i64.get();
		case DType::int16:
			return os << s.data.i16.get();
		case DType::uint16:
			return os << s.data.i16_u.get();
		case DType::int8:
			return os << s.data.i8.get();
		case DType::uint8:
			return os << s.data.i8_u.get();
		case DType::Complex64:
			return os << s.data.c64.get();
		case DType::Complex128:
			return os << s.data.c128.get();
		case DType::Bool:
			return os << s.data.b.get();
		case DType::TensorObj:
			return os << s.data.t.get();
#ifdef __SIZEOF_INT128__
		case DType::int128:
			return os << s.data.i128.get();
		case DType::uint128:
			return os << s.data.i128_u.get();
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return os << s.data.f16.get();
		case DType::Complex32:
			return os << s.data.c32.get();
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return os << s.data.f128.get();
#endif
	}
	return os;
}

}
