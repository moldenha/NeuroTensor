/*
 * Utilities for convert
*/

#ifndef NT_CONVERT_UTILS_H__
#define NT_CONVERT_UTILS_H__

#include "../utils/type_traits.h"
#include "../dtype/compatible/DType_compatible.h"

namespace nt::convert::details{

template<typename T>
inline static constexpr bool valid_convert_type_v = 
    nt::type_traits::is_decay_in_v<T, bool, nt::uint_bool_t> 
    || nt::DTypeFuncs::type_to_dtype<nt::type_traits::decay_t<T>> != nt::DType::Bool;

#define NT_CHECK_VALID_CONVERT_TYPES_MACRO__(type, name_a, name_b)\
    static_assert(valid_convert_type_v<type>, "Error type for " #name_a " is not seen as a valid convert type");

NT_GET_DEFINE_FLOATING_DTYPES_(NT_CHECK_VALID_CONVERT_TYPES_MACRO__)
NT_GET_DEFINE_COMPLEX_DTYPES_(NT_CHECK_VALID_CONVERT_TYPES_MACRO__)
NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CHECK_VALID_CONVERT_TYPES_MACRO__)
NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CHECK_VALID_CONVERT_TYPES_MACRO__)
NT_GET_DEFINE_OTHER_DTYPES_(NT_CHECK_VALID_CONVERT_TYPES_MACRO__)


#undef NT_CHECK_VALID_CONVERT_TYPES_MACRO__ 

}

#endif

