#ifndef __DTYPE_COMPATIBLE_T__
#define __DTYPE_COMPATIBLE_T__

#include "../../types/Types.h"
/* #if defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__) */
/* #include "DType_compatible_all.h" */
/* #elif !defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && !defined(__SIZEOF_INT128__) */ 
/* #include "DType_compatible_float128.h" */
/* #elif !defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__) */
/* #include "DType_compatible_float128_int128.h" */
/* #elif defined(_HALF_FLOAT_SUPPORT_) && !defined(_128_FLOAT_SUPPORT_) && !defined(__SIZEOF_INT128__) */
/* #include "DType_compatible_float16.h" */
/* #elif defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && !defined(__SIZEOF_INT128__) */
/* #include "DType_compatible_float16_float128.h" */
/* #elif defined(_HALF_FLOAT_SUPPORT_) && !defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__) */
/* #include "DType_compatible_float16_int128.h" */
/* #elif !defined(_HALF_FLOAT_SUPPORT_) && !defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__) */
/* #include "DType_compatible_int128.h" */
/* #else */
/* #include "DType_compatible_standard.h" */
/* #endif */
#include "compatible_macro.h"


_NT_REGISTER_FLOATING_TYPE(float, nt::DType::Float32)
_NT_REGISTER_FLOATING_TYPE(double, nt::DType::Float64)
_NT_REGISTER_INTEGER_TYPE(int64_t, nt::DType::int64)
_NT_REGISTER_INTEGER_TYPE(int32_t, nt::DType::int32)
_NT_REGISTER_INTEGER_TYPE(int16_t, nt::DType::int16)
_NT_REGISTER_INTEGER_TYPE(int8_t, nt::DType::int8)
_NT_REGISTER_UNSIGNED_TYPE(uint32_t, nt::DType::uint32)
_NT_REGISTER_UNSIGNED_TYPE(uint16_t, nt::DType::uint16)
_NT_REGISTER_UNSIGNED_TYPE(uint8_t, nt::DType::uint8)
_NT_REGISTER_COMPLEX_TYPE(complex_64, nt::DType::cfloat)
_NT_REGISTER_COMPLEX_TYPE(complex_128, nt::DType::cdouble)
_NT_REGISTER_OTHER_TYPE(uint_bool_t, nt::DType::Bool)
_NT_REGISTER_OTHER_TYPE(Tensor, nt::DType::TensorObj)
#ifdef _HALF_FLOAT_SUPPORT_
_NT_REGISTER_FLOATING_TYPE(float16_t, nt::DType::Float16)
_NT_REGISTER_COMPLEX_TYPE(complex_32, nt::DType::Complex32)
#endif //_HALF_FLOAT_SUPPORT_
#ifdef _128_FLOAT_SUPPORT_
_NT_REGISTER_FLOATING_TYPE(float128_t, nt::DType::Float128)
#endif //_128_FLOAT_SUPPORT_
#ifdef __SIZEOF_INT128__
_NT_REGISTER_INTEGER_TYPE(int128_t, nt::DType::int128)
_NT_REGISTER_UNSIGNED_TYPE(uint128_t, nt::DType::uint128)
#endif //__SIZEOF_INT128__

_NT_REGISTER_ALL_ITERATORS_()

namespace nt{ namespace DTypeFuncs{
_NT_CHECK_FLOATING_(DType::Float32, float)
_NT_CHECK_FLOATING_(DType::Float64, double)
_NT_CHECK_COMPLEX_(DType::Complex64, complex_64)
_NT_CHECK_COMPLEX_(DType::Complex128, complex_128)
_NT_CHECK_OTHER_(DType::Bool, uint_bool_t)
_NT_CHECK_OTHER_(DType::TensorObj, Tensor)
_NT_CHECK_SIGNED_(DType::int8, int8_t)
_NT_CHECK_SIGNED_(DType::int16, int16_t)
_NT_CHECK_SIGNED_(DType::int32, int32_t)
_NT_CHECK_SIGNED_(DType::int64, int64_t)
_NT_CHECK_UNSIGNED_(DType::uint8, uint8_t)
_NT_CHECK_UNSIGNED_(DType::uint16, uint16_t)
_NT_CHECK_UNSIGNED_(DType::uint32, uint32_t)
#ifdef __SIZEOF_INT128__
_NT_CHECK_UNSIGNED_(DType::uint128, uint128_t)
_NT_CHECK_SIGNED_(DType::int128, int128_t)
#endif
#ifdef _HALF_FLOAT_SUPPORT_
_NT_CHECK_FLOATING_(DType::Float16, float16_t)
_NT_CHECK_COMPLEX_(DType::Complex32, complex_32)
#endif
#ifdef _128_FLOAT_SUPPORT_
_NT_CHECK_FLOATING_(DType::Float128, float128_t)
#endif
}}



#endif //__DTYPE_COMPATIBLE_T__
