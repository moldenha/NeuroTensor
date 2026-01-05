// The entire point of this cpp file is to reduce the compile time of NeuroTensor (a little bit)
// by storing the checks of a lot of the type traits and etc. that work by compile time static_asserts in here
// That way, it only needs to be checked once instead of the 100+ times in all the header files it would be included in.


/* this half checks the mathematics functions */
#include "../../math/math.h"
#include "DTypeDeclareMacros.h"

// Requirements:
//  - sqrt(any scalar)
//  - pow(scalar<A>, scalar<A>);
//  - pow(scalar<A>, any integer)


#define NT_CHECK_SINGLE_SCALAR_FUNCTION_HELPER_(type, name_a, name_b, func_name) \
    static_assert( \
        std::is_invocable_r_v< \
            type, \
            decltype(static_cast<type(*)(const type&)>(&::nt::math::func_name)), \
            const type& \
        >, \
        "Error: " #func_name " is not valid for type " #type \
    );

#define NT_CHECK_SINGLE_SCALAR_FUNCTION_(func_name) \
    NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_OTHER_(NT_CHECK_SINGLE_SCALAR_FUNCTION_HELPER_, func_name) \
    NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_OTHER_(NT_CHECK_SINGLE_SCALAR_FUNCTION_HELPER_, func_name) \
    NT_GET_DEFINE_FLOATING_DTYPES_OTHER_(NT_CHECK_SINGLE_SCALAR_FUNCTION_HELPER_, func_name) \
    NT_GET_DEFINE_COMPLEX_DTYPES_OTHER_(NT_CHECK_SINGLE_SCALAR_FUNCTION_HELPER_, func_name)



NT_CHECK_SINGLE_SCALAR_FUNCTION_(sqrt);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(exp);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(log);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(sqrt);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(abs);

NT_CHECK_SINGLE_SCALAR_FUNCTION_(tanh);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(cosh);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(sinh);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(asinh);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(acosh);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(atanh);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(atan);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(asin);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(acos);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(tan);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(sin);
NT_CHECK_SINGLE_SCALAR_FUNCTION_(cos);


// Other Funtion Requirements:
// pow (scalar<A>, scalar<A>);

#define NT_CHECK_POW_FN_COMPLEX_(type, name_a, name_b)                                   \
    static_assert(std::is_invocable_r_v<type, decltype(static_cast<type(*)(const type&, const type&)>(&::nt::math::pow<type::value_type>)), \
                  const type&, const type&>,                                     \
        "Error: nt::math::pow(const " #type "&, const " #type "&) not invocable" \
    );

#define NT_CHECK_POW_FN_FLOATING_(type, name_a, name_b)                                   \
    static_assert(std::is_invocable_r_v<type, decltype(static_cast<type(*)(const type&, const type&)>(&::nt::math::pow<type>)), \
                  const type&, const type&>,                                     \
        "Error: nt::math::pow(const " #type "&, const " #type "&) not invocable" \
    );

#define NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_NON_FLOATING_(exp_t, name_a, name_b, base_t)    \
    static_assert(std::is_invocable_r_v<                                         \
        base_t, decltype(static_cast<base_t(*)(const base_t&, const exp_t&)>(&::nt::math::pow<base_t, exp_t>)),                         \
        const base_t&, const exp_t&>,                                             \
        "Error: nt::math::pow(const " #base_t "&, const " #exp_t "&) not invocable" \
    );


#define NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_FLOATING_(exp_t, name_a, name_b, base_t)    \
    static_assert(std::is_invocable_r_v<                                         \
        base_t, decltype(static_cast<base_t(*)(const base_t&, const exp_t&)>(&::nt::math::pow<exp_t>)), \
        const base_t&, const exp_t&>,                                             \
        "Error: nt::math::pow(const " #base_t "&, const " #exp_t "&) not invocable" \
    );

#define NT_CHECK_POW_FN_RETURN_INTEGER_FLOATING_(type, name_a, name_b)\
    NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_OTHER_(NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_FLOATING_, type)\
    NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_OTHER_(NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_FLOATING_, type)

#define NT_CHECK_POW_FN_RETURN_INTEGER_NON_FLOATING_(type, name_a, name_b)\
    NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_OTHER_(NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_NON_FLOATING_, type)\
    NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_OTHER_(NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_NON_FLOATING_, type)


NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_POW_FN_FLOATING_); 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_CHECK_POW_FN_COMPLEX_); 
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_POW_FN_FLOATING_); 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CHECK_POW_FN_FLOATING_);

NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_POW_FN_RETURN_INTEGER_FLOATING_); 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_CHECK_POW_FN_RETURN_INTEGER_NON_FLOATING_); 
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_POW_FN_RETURN_INTEGER_NON_FLOATING_); 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CHECK_POW_FN_RETURN_INTEGER_NON_FLOATING_); 


#undef NT_CHECK_POW_FN_FLOATING_
#undef NT_CHECK_POW_FN_COMPLEX_
#undef NT_CHECK_POW_FN_RETURN_INTEGER_FLOATING_
#undef NT_CHECK_POW_FN_RETURN_INTEGER_NON_FLOATING_
#undef NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_NON_FLOATING_ 
#undef NT_CHECK_POW_FN_RETURN_INTEGER_HELPER_FLOATING_ 


#undef NT_CHECK_SINGLE_SCALAR_FUNCTION_HELPER_
#undef NT_CHECK_SINGLE_SCALAR_FUNCTION_


#define NT_CHECK_IS_NAN_FN_(type, name_a, name_b)\
    static_assert(std::is_invocable_r_v<bool, decltype(static_cast<bool (*)(const type&)>(::nt::math::isnan)),  \
        const type&>, \
        "Error, isnan with type " #type " is not invocable!"); \

NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_IS_NAN_FN_); 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_CHECK_IS_NAN_FN_); 
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_IS_NAN_FN_); 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CHECK_IS_NAN_FN_); 

#undef NT_CHECK_IS_NAN_FN_

#define NT_CHECK_IS_INF_FN_FLOATING_(type, name_a, name_b)\
    static_assert(::nt::math::isinf(::nt::math::inf<type>()) && ::nt::math::isinf(::nt::math::neg_inf<type>()), \
                "Error, there should be an infinity for every type (pseudo or not)"); \

#define NT_CHECK_IS_INF_FN_(type, name_a, name_b)\
    static_assert(!::nt::math::isinf(::nt::math::inf<type>()) && !::nt::math::isinf(::nt::math::neg_inf<type>()), \
                "Error, there should be an infinity for every type (pseudo or not)"); \


NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_IS_INF_FN_FLOATING_); 
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_CHECK_IS_INF_FN_FLOATING_); 
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_IS_INF_FN_); 
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CHECK_IS_INF_FN_); 

#undef NT_CHECK_IS_INF_FN_
#undef NT_CHECK_IS_INF_FN_FLOATING_

#define NT_CHECK_ROUND_FN__(type, name_a, name_b, func_name)\
    static_assert(std::is_invocable_r_v<type, decltype(static_cast<type (*)(const type&)>(::nt::math::func_name)),  \
        const type&>, \
        "Error, " #func_name " with type " #type " is not invocable!"); \

#define NT_CHECK_ROUND_FN_(func_name)\
NT_GET_DEFINE_FLOATING_DTYPES_OTHER_(NT_CHECK_ROUND_FN__, func_name); \
NT_GET_DEFINE_COMPLEX_DTYPES_OTHER_(NT_CHECK_ROUND_FN__, func_name); \

NT_CHECK_ROUND_FN_(trunc)
NT_CHECK_ROUND_FN_(round)
NT_CHECK_ROUND_FN_(ceil)
NT_CHECK_ROUND_FN_(floor)

#undef NT_CHECK_ROUND_FN_
#undef NT_CHECK_ROUND_FN__


// This is going to check the DTypeFuncs::
// You can think of DTypeFuncs:: as the type_traits:: of constexpr DType's


#include "../../types/Types.h"
#include "compatible_macro.h"
#include "DType_compatible.h"

namespace nt::DTypeFuncs{
#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_CHECK_FLOATING_(DType::dtype_enum_a, type)
NT_GET_X_FLOATING_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_CHECK_COMPLEX_(DType::dtype_enum_a, type)
NT_GET_X_COMPLEX_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_CHECK_SIGNED_(DType::dtype_enum_a, type)
NT_GET_X_SIGNED_INTEGER_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_CHECK_UNSIGNED_(DType::dtype_enum_a, type)
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
#undef X

#define X(type, dtype_enum_a, dtype_enum_b)\
    NT_CHECK_OTHER_(DType::dtype_enum_a, type)
NT_GET_X_OTHER_DTYPES_
#undef X

}
