#ifndef NT_MATH_FLOAT128_MATH_FUNCTIONS_H__
#define NT_MATH_FLOAT128_MATH_FUNCTIONS_H__
// this is a header file for general functions to be implemented

/*
This header file is meant to act as a standard math functions for 
 all supported types to run on the cpu for scalars


nt::math::sqrt(T)
nt::math::exp(T)
nt::math::log(T)
nt::math::abs(T)

nt::math::tanh(T)
nt::math::cosh(T)
nt::math::sinh(T)
nt::math::asinh(T)
nt::math::acosh(T)
nt::math::atanh(T)
nt::math::atan(T)
nt::math::asin(T)
nt::math::acos(T)
nt::math::tan(T)
nt::math::sin(T)
nt::math::cos(T)

nt::math::abs(T)

nt::math::isnan(T) (only floating and complex really valid)
nt::math::isinf(T) (only floating and complex really valid)
nt::math::nan<T>() (integers return 0)
nt::math::inf<T>() (integers return max)
nt::math::neg_inf<T>() (integers return min)

nt::math::trunc(T)
nt::math::round(T)
nt::math::ceil(T)
nt::math::floor(T)

nt::math::pow(T, (T or Integer)

nt::math::fmod(floating, floating)

*/


#include "math/abs.hpp"
#include "math/acos.hpp"
#include "math/acosh.hpp"
#include "math/asin.hpp"
#include "math/asinh.hpp"
#include "math/atan.hpp"
#include "math/atanh.hpp"
#include "math/ceil.hpp"
#include "math/cos.hpp"
#include "math/cosh.hpp"
#include "math/exp.hpp"
#include "math/floor.hpp"
#include "math/fmod.hpp"
#include "math/inf.hpp"
#include "math/log.hpp"
#include "math/nan.hpp"
#include "math/pow.hpp"
#include "math/round.hpp"
#include "math/sin.hpp"
#include "math/sinh.hpp"
#include "math/sqrt.hpp"
#include "math/tan.hpp"
#include "math/tanh.hpp"
#include "math/trunc.hpp"





#endif
