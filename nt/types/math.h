// this is a header file for general functions to be implemented
#include <cmath>
#include "float16.h"
#include "float128.h"
#include "../dtype/compatible/DTypeDeclareMacros.h"
#include "../convert/Convert.h"
#include "../utils/always_inline_macro.h"
#include "../utils/type_traits.h"
#include "../dtype/compatible/DType_compatible.h"

namespace nt{
namespace scalar_math_details{

template<DType dt>
struct integer_in_floating_out{
    using t = typename std::conditional<
        DTypeFuncs::is_convertible_to_floating<dt>,
        DTypeFuncs::dtype_to_type_t<DTypeFuncs::convert_to_floating<dt>>,
        float
    >::type;
};

#define X(type, dtna, dtnb) \
NT_ALWAYS_INLINE typename integer_in_floating_out<DType::dtna>::t \
integer_to_floating(const type& i) { \
    using out_type = typename integer_in_floating_out<DType::dtna>::t; \
    return convert::convert<out_type>(i); \
}

NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_  
#undef X

}

#define NT_FUNCTIONAL_CONVERT_FUNC_CONVERT(my_type, other_type, function, name)\
    convert::convert<my_type>(function(convert::convert<other_type>(name)))


// this is a macro meant to be used after all the floating type definitions
// have been called
// look at the sqrt use for an example
#define NT_FUNCTIONAL_MASS_INTEGER_CONVERT(type, a, b, func_name)\
NT_ALWAYS_INLINE type func_name(const type& i){\
    return convert::convert<type>(func_name(scalar_math_details::integer_to_floating(i)));\
}

// SQRT
NT_ALWAYS_INLINE ::nt::float16_t sqrt(const ::nt::float16_t& f){
    return NT_FUNCTIONAL_CONVERT_FUNC_CONVERT(::nt::float16_t, float, std::sqrt, f);
    return convert::convert<::nt::float16_t>(
            std::sqrt(convert::convert<float>(f)));
}

NT_ALWAYS_INLINE float sqrt(const float& f){
    return std::sqrt(f);
}

NT_ALWAYS_INLINE double sqrt(const double& f){
    return std::sqrt(f);
}

NT_ALWAYS_INLINE ::nt::float128_t sqrt(const ::nt::float128_t& f){
    return NT_FUNCTIONAL_CONVERT_FUNC_CONVERT(::nt::float128_t, double, std::sqrt, f);
}

#define X(type, a, b)\
NT_ALWAYS_INLINE type sqrt(const type& f){\
    return type(sqrt(f.real()), sqrt(f.imag()));\
}
NT_GET_X_COMPLEX_DTYPES_
#undef X

// this is a macro to make the sqrt function for all integer types
#define NT_CUR_FUNC__(type, a, b)\
    NT_FUNCTIONAL_MASS_INTEGER_CONVERT(type, a, b, sqrt)

NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CUR_FUNC__);
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CUR_FUNC__);

#undef NT_CUR_FUNC__

//SQRT END

#undef NT_FUNCTIONAL_CONVERT_FUNC_CONVERT 
#undef NT_FUNCTIONAL_MASS_INTEGER_CONVERT 
}
