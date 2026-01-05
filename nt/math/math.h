#ifndef NT_MATH_MATH_FUNCTIONS_H__
#define NT_MATH_MATH_FUNCTIONS_H__
// this is a header file for general functions to be implemented

/*
This header file is meant to act as a standard math functions for 
 all supported types to run on the cpu for scalars

*NOTE* You will notice not all std math functions have been implemented
          only certain ones are needed, the ones that are needed have been implemented
          the math functions, and this header file in general, are meant to serve as a more
          as-needed basis


Even for user use cases, if a user wants to use nt::float16_t for example, there is gurenteed
    Support for the math functions below:

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


#include "functional/abs.hpp"
#include "functional/acos.hpp"
#include "functional/acosh.hpp"
#include "functional/asin.hpp"
#include "functional/asinh.hpp"
#include "functional/atan.hpp"
#include "functional/atanh.hpp"
#include "functional/ceil.hpp"
#include "functional/cos.hpp"
#include "functional/cosh.hpp"
#include "functional/exp.hpp"
#include "functional/floor.hpp"
#include "functional/fmod.hpp"
#include "functional/inf.hpp"
#include "functional/log.hpp"
#include "functional/nan.hpp"
#include "functional/pow.hpp"
#include "functional/round.hpp"
#include "functional/sin.hpp"
#include "functional/sinh.hpp"
#include "functional/sqrt.hpp"
#include "functional/tan.hpp"
#include "functional/tanh.hpp"
#include "functional/trunc.hpp"
#include "../types/float128/math.h" // float128 math


#define NT_UNDEF_MATH_UTIL_MACROS
#include "functional/utils.h"
#undef NT_UNDEF_MATH_UTIL_MACROS


#endif
