#include "../Tensor.h"
#include "Scalar.h"
#include "../types/Types.h"
#include "../math/math.h"
#include "../convert/Convert.h"
#include "DType_enum.h"
#include <limits>
#include <cmath>

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
        v.d = math::inf<decltype(v.d)>();
    }else if(_str == "-inf"){
        v.d = math::neg_inf<decltype(v.d)>();
    }
    else if(_str == "nan"){
        v.d = math::nan<decltype(v.d)>();
    }
    else {
        // Try to convert string to a number
        try {
            v.d = (::nt::float128_func::from_string(_str));  // Convert the string to a double
        }
        catch (const std::invalid_argument& e) {
            // If conversion fails, throw an exception
            utils::throw_exception(false, "Unsupported string to scalar: $ (reason: $)", _str, e.what());
        }
    }
}


bool Scalar::isInfinity() const {
    if(isFloatingPoint()){
        return math::isinf(v.d);
    }else if(isComplex()){
        return math::isinf(v.c);
    }else if(isIntegral()){
        return math::isinf(v.i);
    }
    return false;

}


bool Scalar::isNan() const {
    if(isFloatingPoint()){
        return math::isnan(v.d);
    }else if(isComplex()){
        return math::isnan(v.c);
    }else if(isIntegral()){
        return math::isnan(v.i);
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
            return Scalar(math::neg_inf<decltype(v.c)>()); 
        }
        return Scalar(math::inf<decltype(v.c)>()); 
    }
    else if(isNan()){
        return Scalar(math::nan<decltype(v.c)>()); 
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
            return Scalar(math::neg_inf<decltype(v.i)>()); 
        }
        return Scalar(math::inf<decltype(v.i)>()); 
    }
    else if(isNan()){
        return Scalar(math::nan<decltype(v.i)>()); 
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
            return Scalar(math::neg_inf<decltype(v.d)>()); 
        }
        return Scalar(math::inf<decltype(v.d)>()); 
    }
    else if(isNan()){
        return Scalar(math::nan<decltype(v.d)>()); 
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
                return math::neg_inf<T>();
            }
        }
        return math::inf<T>();
    }
    else if(isNan()){
        return math::nan<T>();
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
            return math::neg_inf<T>();
        }
        return math::inf<T>();
    }
    else if(isNan()){
        return math::nan<T>();
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
            return math::neg_inf<T>();
        }
        return math::inf<T>();
    }
    else if(isNan()){
        // std::cout << "self is nan" << std::endl;
        return math::nan<T>();
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
