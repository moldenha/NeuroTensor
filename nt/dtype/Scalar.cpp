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
    else if constexpr (std::is_same_v<T, ::nt::int128_t>){
        return false;
    }
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

#define X(type, dtype_enum_a, dtype_enum_b)\
template void Scalar::init_from<type>(const type&);
NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_

#undef X

template void Scalar::init_from<bool>(const bool&);
template void Scalar::init_from<uint_bool_t>(const uint_bool_t&);


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
        return Scalar(decltype(v.c)(convert::convert<decltype(v.c)::value_type>(v.d), convert::convert<decltype(v.c)::value_type>(v.d)));
    }else if(isIntegral()){
        return Scalar(decltype(v.c)(convert::convert<decltype(v.c)::value_type>(v.i), convert::convert<decltype(v.c)::value_type>(v.i)));
    }
    return Scalar(decltype(v.c)(convert::convert<decltype(v.c)::value_type>(v.i), convert::convert<decltype(v.c)::value_type>(v.i)));
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
#ifndef BOOST_MP_STANDALONE
		return DType::Float128;
#else
		return DType::Double;
#endif
	}
	else if(isIntegral()){
		return DType::int128;
	}
	else if(isBoolean())
		return DType::Bool;
	else
		throw std::runtime_error("Unknown scalar type");
}


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


// By default complex floating numbers to complex numbers when it comes to Scalar type
// instead of (val, 0)
// The reason for this is if a user does tensor /= 1;
// (Example)
// and then if the tensor is complex that will lead to division by 0
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
	else if(isFloatingPoint()){
		return T(convert::convert<typename T::value_type>(v.d), convert::convert<typename T::value_type>(v.d));
    }
    return T(convert::convert<typename T::value_type>(v.i), convert::convert<typename T::value_type>(v.i));
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

#define X(type, dtype_enum_a, dtype_enum_b)\
template type Scalar::to<type>() const;

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_

#undef X

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



}
