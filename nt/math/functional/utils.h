/*
 * These are some general utilities that make making each mathematics function easier and
 * less time consuming
*/


#ifndef NT_MATH_FUNCTIONAL_UTILITIES_H__
#define NT_MATH_FUNCTIONAL_UTILITIES_H__ 

#ifndef NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__ 
#ifndef _NT_EXPAND_
#define _NT_EXPAND_(x) x
#endif

#ifndef NT_GLUE_EXPAND
#define NT_GLUE_EXPAND(x, y) _NT_EXPAND_(x)y
#endif

#ifndef NT_STRINGIFY
#define NT_STRINGIFY__(x) #x
#define NT_STRINGIFY(x) NT_STRINGIFY__(x)
#endif

// # define P_INCLUDE_FILE P_XSTR(P_CONCAT(Fonts/,P_CONCAT(LED_FONT,.h)))
// # include P_INCLUDE_FILE

#define NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(name) NT_STRINGIFY(NT_GLUE_EXPAND(../../types/float128/math/, NT_GLUE_EXPAND(name, .hpp)))
#endif

#include "../../types/bit_128_integer.h"
#include "../../types/float128.h"
#include "../../utils/type_traits.h"
#include "../../dtype/compatible/DType_compatible.h"
#include "../../convert/Convert.h"
#include "../../utils/always_inline_macro.h"
#include "../../bit/float_bits.h"
#include <cmath>

// utilities for generalizing math functions
namespace nt::math::scalar_math_details{

template<DType dt>
struct integer_in_floating_out{
    using t = typename ::nt::type_traits::conditional<
        DTypeFuncs::is_convertible_to_floating<dt>,
        DTypeFuncs::dtype_to_type_t<DTypeFuncs::convert_to_floating<dt>>,
        float
    >::type;
};





#define NT_INTEGER_IN_FLOATING_OUT_FUNC_(type, dtna, dtnb) \
NT_ALWAYS_INLINE integer_in_floating_out<DType::dtna>::t \
    integer_to_floating(const type& i){\
    using out_type = typename integer_in_floating_out<DType::dtna>::t;\
    return convert::convert<out_type>(i);\
}\

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_INTEGER_IN_FLOATING_OUT_FUNC_)
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_INTEGER_IN_FLOATING_OUT_FUNC_)
#undef NT_INTEGER_IN_FLOATING_OUT_FUNC_


}

#define NT_MAKE_SINGULAR_FUNCTION_COMPLEX_(type, name_a, name_b, func_name)\
NT_ALWAYS_INLINE type func_name(const type& a){\
    return type(func_name(a.real()), func_name(a.imag()));\
}

#define NT_MAKE_SINGULAR_FUNCTION_INTEGER_(type, name_a, name_b, func_name) \
NT_ALWAYS_INLINE type func_name(const type& a){\
    using conv_t = type_traits::conditional_t< \
            ::nt::DTypeFuncs::is_convertible_to_floating<DType::name_a>, \
               ::nt::DTypeFuncs::dtype_to_type_t< \
                    ::nt::DTypeFuncs::convert_to_floating<DType::name_a> \
                >, \
            float>;\
    return ::nt::convert::convert<type>(\
            static_cast<conv_t(*)(const conv_t&)>(&::nt::math::func_name)(::nt::convert::convert<conv_t>(a))\
        );\
}

#ifdef SIMDE_FLOAT16_IS_SCALAR
#define NT_FLOAT16_MAKE_SINGULAR_FUNCTION(func_name)\
NT_ALWAYS_INLINE ::nt::float16_t func_name(const ::nt::float16_t& a){\
    return ::nt::convert::convert<::nt::float16_t>(func_name(::nt::convert::convert<float>(a)));\
}

#else
#define NT_FLOAT16_MAKE_SINGULAR_FUNCTION(func_name)\
NT_ALWAYS_INLINE ::nt::float16_t func_name(const ::nt::float16_t& a){\
    half_float::half(half_float::detail::func_name(a))\
}

#endif


// this includes the std::func_name for float and double
// From there, it includes the float16 implementation, and then it uses those definitions
// To make the complex type, and integer types
// Afterwards, you need to manually do #include NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__(name) -> <nt/types/float128/math/func_name.hpp>

#define NT_MAKE_SINGULAR_FUNCTION_(func_name)\
namespace nt::math{\
NT_ALWAYS_INLINE float func_name(const float& a){\
    return std::func_name(a);\
}\
NT_ALWAYS_INLINE double func_name(const double& a){\
    return std::func_name(a);\
}\
NT_FLOAT16_MAKE_SINGULAR_FUNCTION(func_name)\
NT_GET_DEFINE_COMPLEX_DTYPES_OTHER_(NT_MAKE_SINGULAR_FUNCTION_COMPLEX_, func_name)\
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_OTHER_(NT_MAKE_SINGULAR_FUNCTION_INTEGER_, func_name)\
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_OTHER_(NT_MAKE_SINGULAR_FUNCTION_INTEGER_, func_name)\
}\



#elif defined(NT_UNDEF_MATH_UTIL_MACROS)
    // undefine macros here
    #ifdef NT_MAKE_SINGULAR_FUNCTION_COMPLEX_
        #undef NT_MAKE_SINGULAR_FUNCTION_COMPLEX_
    #endif
    #ifdef NT_MAKE_SINGULAR_FUNCTION_INTEGER_
        #undef NT_MAKE_SINGULAR_FUNCTION_INTEGER_
    #endif
    #ifdef NT_MAKE_SINGULAR_FUNCTION_
        #undef NT_MAKE_SINGULAR_FUNCTION_
    #endif
    #ifdef NT_FLOAT16_MAKE_SINGULAR_FUNCTION
        #undef NT_FLOAT16_MAKE_SINGULAR_FUNCTION
    #endif
    #ifdef NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__
        #undef NT_MAKE_FLOAT128_MATH_FUNCTION_ACCESSIBLE__
    #endif
#endif
