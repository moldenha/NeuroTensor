#include "../Tensor.h"
#include "Scalar.h"
#include "../types/Types.h"
#include "../convert/Convert.h"
#include "DType_enum.h"
#include <limits>
#include <cmath>

#include <type_traits>

namespace nt{

namespace details{
template<typename T>
inline constexpr T get_infinity(){
    if constexpr (std::is_same_v<T, complex_32>){
        return complex_32(get_infinity<float16_t>(), get_infinity<float16_t>());
    }
    else if constexpr(std::is_same_v<T, complex_64>){
        return complex_64(get_infinity<float>(), get_infinity<float>());
    }
    else if constexpr(std::is_same_v<T, complex_128>){
        return complex_128(get_infinity<double>(), get_infinity<double>());
    }
    else if constexpr (std::numeric_limits<T>::has_infinity){
        return std::numeric_limits<T>::infinity();
    }
    else{
        return std::numeric_limits<T>::max();
    }
}

template<typename T>
inline bool is_inf(T val){
    if constexpr (std::is_same_v<T, complex_32>){
        return is_inf(val.real()) && is_inf(val.imag());
    }
    else if constexpr(std::is_same_v<T, complex_64>){
        return is_inf(val.real()) && is_inf(val.imag());
    }
    else if constexpr(std::is_same_v<T, complex_128>){
        return is_inf(val.real()) && is_inf(val.imag());
    }
    else if constexpr (std::numeric_limits<T>::has_infinity){
        return std::isinf(val) || std::isinf(-val);
    }
    else{
        return std::numeric_limits<T>::max() == val || std::numeric_limits<T>::max() == (-val);
    }
}

template<typename T>
inline constexpr T get_nan(){
    if constexpr (std::is_same_v<T, complex_32>){
        return complex_32(get_nan<float16_t>(), get_nan<float16_t>());
    }
    else if constexpr(std::is_same_v<T, complex_64>){
        return complex_64(get_nan<float>(), get_nan<float>());
    }
    else if constexpr(std::is_same_v<T, complex_128>){
        return complex_128(get_nan<double>(), get_nan<double>());
    }
    else if constexpr(std::is_same_v<T, float>){
        return std::nanf("");  
    }else if constexpr (std::is_same_v<T, double>){
        return std::nan("");
    }else if constexpr (std::is_same_v<T, long double>){
        return std::nanl("");
    }
#ifndef SIMDE_FLOAT16_IS_SCALAR
    else if constexpr(std::is_same_v<T, float16_t>){
        return half_float::nanh("");
    }
#else
    else if constexpr(std::is_same_v<T, float16_t>){
        return _NT_FLOAT32_TO_FLOAT16_(std::nanf(""));
    }
#endif
    else{
        return static_cast<T>(std::nanl(""));
    }
}

template<typename T>
inline bool is_nan(T val){
    if constexpr (std::is_same_v<T, complex_32>){
        return is_nan(val.real()) && is_inf(val.imag());
    }
    else if constexpr(std::is_same_v<T, complex_64>){
        return is_nan(val.real()) && is_inf(val.imag());
    }
    else if constexpr(std::is_same_v<T, complex_128>){
        return is_nan(val.real()) && is_inf(val.imag());
    }else if constexpr(std::is_same_v<T, float16_t>){
        return val != val; //for nan values they are not equal to themselves
    }
#ifdef __SIZEOF_INT128__
    else if constexpr (std::is_same_v<T, ::nt::int128_t>){
        return false;
    }
#endif
    else{
        return std::isnan(val);    
    }
}

}

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

template<typename T>
void Scalar::init_from(const T& vv){
    using namespace DTypeFuncs;
    this->dtype = type_to_dtype<T>; 
    if constexpr (std::is_same_v<T, bool>){
        dtype = DType::Bool;
        v.i = vv ? 1 : 0;
    }else if constexpr(std::is_same_v<T, uint_bool_t>){
        dtype = DType::Bool;
        v.i = (vv == true) ? 1 : 0;
    }else if constexpr(is_dtype_floating_v<type_to_dtype<T>>){
        v.d = convert::convert<decltype(v.d)>(vv);
    }else if constexpr(is_dtype_integer_v<type_to_dtype<T>>){
        v.i = convert::convert<decltype(v.i)>(vv);
    }else if constexpr(is_dtype_complex_v<type_to_dtype<T>>){
        v.c = convert::convert<decltype(v.c)>(vv);
    }
}

template void Scalar::init_from<bool>(const bool&);
template void Scalar::init_from<uint_bool_t>(const uint_bool_t&);
template void Scalar::init_from<int8_t>(const int8_t&);
template void Scalar::init_from<uint8_t>(const uint8_t&);
template void Scalar::init_from<int16_t>(const int16_t&);
template void Scalar::init_from<uint16_t>(const uint16_t&);
template void Scalar::init_from<int32_t>(const int32_t&);
template void Scalar::init_from<uint32_t>(const uint32_t&);
template void Scalar::init_from<int64_t>(const int64_t&);
#ifdef __SIZEOF_INT128__
template void Scalar::init_from<int128_t>(const int128_t&);
template void Scalar::init_from<uint128_t>(const uint128_t&);
#endif
template void Scalar::init_from<float16_t>(const float16_t&);
template void Scalar::init_from<float>(const float&);
template void Scalar::init_from<double>(const double&);
template void Scalar::init_from<float128_t>(const float128_t&);
template void Scalar::init_from<complex_32>(const complex_32&);
template void Scalar::init_from<complex_64>(const complex_64&);
template void Scalar::init_from<complex_128>(const complex_128&);


// template<>
// void Scalar::init_from<int32_t>(const int32_t& vv){
//     dtype = DType::Integer; 
//     v.i = convert::convert<decltype(v.i)>(vv);
// } 

// #ifdef __SIZEOF_INT128__
// template<>
// void Scalar::init_from<int128_t>(const int128_t& vv){
// 	dtype = DType::int128;
//     v.i = convert::convert<decltype(v.i)>(vv);
// } 

// template<>
// void Scalar::init_from<uint128_t>(uint128_t vv){
//     dtype = DType::uint128;
//     v.i = convert::convert<decltype(v.i)>(vv);
// } 
// #endif
// template <>
// void Scalar::init_from<float16_t>(float16_t vv){
//     dtype = DType::Float16;
//     v.d = convert::convert<decltype(v.d)>(vv);
// }

// template <>
// void Scalar::init_from<complex_32>(complex_32 vv){
//     dtype = DType::Complex32;
//     v.c = convert::convert<decltype(v.c)>(vv);
// }
// template <>
// void Scalar::init_from<float128_t>(float128_t vv){
//     dtype = DType::Float128;
//     v.d = convert::convert<decltype(v.d)>(vv);
// }
// template <>
// void Scalar::init_from<double>(double vv){
//     dtype = DType::Double;
//     v.d = convert::convert<decltype(v.d)>(vv);
// }
// template <>
// void Scalar::init_from<float>(float vv){
//     dtype = DType::Float;
//     v.d = convert::convert<decltype(v.d)>(vv);
// }
// template <>
// void Scalar::init_from<uint32_t>(uint32_t vv){
//     dtype = DType::uint32;
//     v.i = convert::convert<decltype(v.i)>(vv);
// }
// template <>
// void Scalar::init_from<complex_64>(complex_64 vv){
//     dtype = DType::Complex64;
//     v.c = convert::convert<decltype(v.c)>(vv);
// }
// template <>
// void Scalar::init_from<complex_128>(complex_128 vv){
//     dtype = DType::Complex128;
//     v.c = convert::convert<decltype(v.c)>(vv);
// }
// template <>
// void Scalar::init_from<uint8_t>(uint8_t vv){
//     dtype = DType::uint8;
//     v.i = convert::convert<decltype(v.i)>(vv);
// }
// template <>
// void Scalar::init_from<int8_t>(int8_t vv){
//     dtype = DType::int8;
//     v.i = convert::convert<decltype(v.i)>(vv);
// }
// template <>
// void Scalar::init_from<int16_t>(int16_t vv){
//     dtype = DType::int16;
//     v.i = convert::convert<decltype(v.i)>(vv);
// }
// template <>
// void Scalar::init_from<uint16_t>(uint16_t vv){
//     dtype = DType::uint16;
//     v.i = convert::convert<decltype(v.i)>(vv);
// }
// template <>
// void Scalar::init_from<int64_t>(int64_t vv){
//     dtype = DType::int64;
//     v.i = convert::convert<decltype(v.i)>(vv);
// } 
// template <>
// void Scalar::init_from<uint_bool_t>(uint_bool_t vv){
//     dtype = DType::Bool;
//     v.i = (vv == true) ? 1 : 0;
// }
// template <>
// void Scalar::init_from<bool>(bool vv){
//     dtype = DType::Bool;
//     v.i = vv ? 1 : 0;
// }

Scalar::Scalar(std::string _str)
    :dtype(DType::Float64)
{
    if(_str == "inf"){
        v.d = details::get_infinity<decltype(v.d)>();
    }else if(_str == "-inf"){
        v.d = -details::get_infinity<decltype(v.d)>();
    }
    else if(_str == "nan"){
        v.d = details::get_nan<decltype(v.d)>();
    }
    else {
        // Try to convert string to a number
        try {
            v.d = std::stod(_str);  // Convert the string to a double
        }
        catch (const std::invalid_argument& e) {
            // If conversion fails, throw an exception
            utils::throw_exception(false, "Unsupported string to scalar: $ (reason: $)", _str, e.what());
        }
    }
}

// Scalar Scalar::toSameType(Scalar s) const {
//     if(s.dtype == dtype) return *this;
//     if(s.isComplex())
//         return toComplex();
//     if(s.isIntegral())
//         return toIntegral();
//     if(s.isFloatingPoint())
//         return toFloatingPoint();
//     return toBoolean();
// }

bool Scalar::isInfinity() const {
    if(isFloatingPoint()){
        return details::is_inf(v.d);
    }else if(isComplex()){
        return details::is_inf(v.c);
    }else if(isIntegral()){
        return details::is_inf(v.i);
    }
    return false;

}


bool Scalar::isNan() const {
    if(isFloatingPoint()){
        return details::is_nan(v.d);
    }else if(isComplex()){
        return details::is_nan(v.c);
    }else if(isIntegral()){
        return details::is_nan(v.i);
    }
    return false;
}



inline Scalar to_same_type(const Scalar& self, const Scalar& to){
    if(self.type() == to.type()) return self;
    if(to.isComplex()) return self.toComplex();
    if(to.isIntegral()) return self.toIntegral();
    if(to.isFloatingPoint()) return self.toFloatingPoint();
    return self.toBoolean();
}



bool Scalar::isEqual(Scalar s) const {

    s = to_same_type(s, *this);
    if(isInfinity()){
        if(isNegative()) return s.isInfinity() && s.isNegative();
        return s.isInfinity() && !s.isNegative();
    }if(isNan()){
        return s.isNan();
    }

    // s = s.toSameType(*this);
    if(isComplex()){
        return v.c == s.v.c;
    }
    if(isFloatingPoint())
        return v.d == s.v.d;
    if(isIntegral())
        return v.i == s.v.i;
    return v.i == s.v.i;
}

Scalar Scalar::operator-(const Scalar& _other) const {
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        return Scalar(v.c - other.v.c);
    }else if(isIntegral()){
        return Scalar(v.i - other.v.i);
    }else if(isFloatingPoint()){
        return Scalar(v.d - other.v.d);
    }
    return *this;
}

Scalar Scalar::operator+(const Scalar& _other) const {
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        return Scalar(v.c + other.v.c);
    }else if(isIntegral()){
        return Scalar(v.i + other.v.i);
    }else if(isFloatingPoint()){
        return Scalar(v.d + other.v.d);
    }
    return *this;
}

Scalar Scalar::operator/(const Scalar& _other) const {
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        return Scalar(v.c / other.v.c);
    }else if(isIntegral()){
        return Scalar(v.i / other.v.i);
    }else if(isFloatingPoint()){
        return Scalar(v.d / other.v.d);
    }
    return *this;
}


Scalar Scalar::operator*(const Scalar& _other) const {
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        return Scalar(v.c * other.v.c);
    }else if(isIntegral()){
        return Scalar(v.i * other.v.i);
    }else if(isFloatingPoint()){
        return Scalar(v.d * other.v.d);
    }
    return *this;
}


Scalar& Scalar::operator-=(const Scalar& _other){
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        v.c -= other.v.c;
    }else if(isIntegral()){
        v.i -= other.v.i;
    }else if(isFloatingPoint()){
        v.d -= other.v.d;
    }
    return *this;
}

Scalar& Scalar::operator+=(const Scalar& _other){
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        v.c += other.v.c;
    }else if(isIntegral()){
        v.i += other.v.i;
    }else if(isFloatingPoint()){
        v.d += other.v.d;
    }
    return *this;
}

Scalar& Scalar::operator/=(const Scalar& _other){
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        v.c /= other.v.c;
    }else if(isIntegral()){
        v.i /= other.v.i;
    }else if(isFloatingPoint()){
        v.d /= other.v.d;
    }
    return *this;
}


Scalar& Scalar::operator*=(const Scalar& _other){
    Scalar other = to_same_type(_other, *this);
    // other = other.toSameType(*this);
    // other = other.toSameType(*this);
    utils::throw_exception(!isBoolean(),
                           "Cannot perform scalar operations on bools");
    if(isComplex()){
        v.c *= other.v.c;
    }else if(isIntegral()){
        v.i *= other.v.i;
    }else if(isFloatingPoint()){
        v.d *= other.v.d;
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




Scalar Scalar::toComplex() const {
    if(isComplex()) return *this;
    else if(isInfinity()){
        if(isNegative()){
            return Scalar(-details::get_infinity<decltype(v.c)>()); 
        }
        return Scalar(details::get_infinity<decltype(v.c)>()); 
    }
    else if(isNan()){
        return Scalar(details::get_nan<decltype(v.c)>()); 
    }
    else if(isFloatingPoint()){
        return Scalar(convert::convert<decltype(v.c)>(v.d));
    }else if(isIntegral()){
        return Scalar(convert::convert<decltype(v.c)>(v.i));
    }
    return Scalar(convert::convert<decltype(v.c)>(v.i));
}

Scalar Scalar::toIntegral() const{
    if(isIntegral()) return *this;
    else if(isInfinity()){
        if(isNegative()){
            return Scalar(-details::get_infinity<decltype(v.i)>()); 
        }
        return Scalar(details::get_infinity<decltype(v.i)>()); 
    }
    else if(isNan()){
        return Scalar(details::get_nan<decltype(v.i)>()); 
    }
    else if(isComplex()){
        return Scalar(convert::convert<decltype(v.i)>(v.c));
    }
    else if(isFloatingPoint()){
        return Scalar(convert::convert<decltype(v.i)>(v.d));
    }
    return Scalar(v.i);
}
Scalar Scalar::toFloatingPoint() const{
    if(isFloatingPoint()) return *this;
    else if(isInfinity()){
        if(isNegative()){
            return Scalar(-details::get_infinity<decltype(v.d)>()); 
        }
        return Scalar(details::get_infinity<decltype(v.d)>()); 
    }
    else if(isNan()){
        return Scalar(details::get_nan<decltype(v.d)>()); 
    }
    else if(isComplex()){
        return Scalar(convert::convert<decltype(v.d)>(v.c));
    }
    else if(isIntegral()){
        return Scalar(convert::convert<decltype(v.d)>(v.i));
    }
    return Scalar(convert::convert<decltype(v.d)>(v.i));
}

Scalar Scalar::toBoolean() const{
    if(isFloatingPoint()){
        return Scalar(convert::convert<uint_bool_t>(v.d));
    }
    else if(isComplex()){
        return Scalar(convert::convert<uint_bool_t>(v.c));
    }
    return Scalar(convert::convert<uint_bool_t>(v.c));
}


bool Scalar::isComplex() const{return DTypeFuncs::is_complex(dtype);}
bool Scalar::isFloatingPoint() const{return DTypeFuncs::is_floating(dtype);}
bool Scalar::isIntegral() const{return DTypeFuncs::is_integer(dtype);}
bool Scalar::isBoolean() const{return dtype == DType::Bool;}
bool Scalar::isZero() const {
	if(isComplex()){
		return v.c == complex_128(0,0);
	}
	else if(isFloatingPoint()){
		return v.d == 0;
	}
	else if(isIntegral()){
		return v.i == 0;
	}
	return v.i == 0;
}

bool Scalar::isNegative() const {
	if(isComplex()){
		return std::get<0>(v.c) < 0 && std::get<1>(v.c) < 0; //both have to be less than zero for negative status
	}else if(isFloatingPoint()){
		return v.d < 0;
	}else if(isIntegral()){
		return v.i < 0;
	}
	return v.i < 0;
}

Scalar Scalar::operator-() const {
	if(isComplex()){
		return Scalar(-v.c);
	}
	else if(isFloatingPoint()){
		return Scalar(-v.d);
	}
	else if(isIntegral()){
		return Scalar(-v.i);
	}
	return Scalar(-v.i);

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
    if(isInfinity()){
        if constexpr (!std::is_unsigned_v<T>){
            if(isNegative()){
                return -details::get_infinity<T>();
            }
        }
        return details::get_infinity<T>();
    }
    else if(isNan()){
        return details::get_nan<T>();
    }
    else if(isComplex())
		return convert::convert<T>(v.c);
	else if(isFloatingPoint())
		return convert::convert<T>(v.d);
	return static_cast<T>(v.i);
}

template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_complex_v<DTypeFuncs::type_to_dtype<T>>, bool>>
T Scalar::to() const{
    if(isInfinity()){
        if(isNegative()){
            return -details::get_infinity<T>();
        }
        return details::get_infinity<T>();
    }
    else if(isNan()){
        return details::get_nan<T>();
    }
    else if(isComplex())
		return T(v.c.real(), v.c.imag());
	else if(isFloatingPoint())
		return convert::convert<T>(v.d);
	return convert::convert<T>(v.i);
}


template<typename T, std::enable_if_t<DTypeFuncs::is_dtype_floating_v<DTypeFuncs::type_to_dtype<T>>, bool>>
T Scalar::to() const{
    if(isInfinity()){
        if(isNegative()){
            return -details::get_infinity<T>();
        }
        return details::get_infinity<T>();
    }
    else if(isNan()){
        return details::get_nan<T>();
    }
    else if(isComplex())
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
template uint_bool_t Scalar::to<bool>() const;


Scalar Scalar::inverse() const{
	if(isComplex()){
		return Scalar(v.c.inverse());
	}
	if(isFloatingPoint()){
		return Scalar((1.0/v.d));
	}
	if(isIntegral()){
		return Scalar(1.0/(double)(v.i));
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
    if(s.isInfinity()){
        if(s.isNegative())
            return os << "-inf";
        return os << "inf";
    }
    if(s.isNan())
        return os << "nan";
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
