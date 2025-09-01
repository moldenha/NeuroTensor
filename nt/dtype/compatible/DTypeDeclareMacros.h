#ifndef NT_DTYPE_DECLARE_MACROS_H__
#define NT_DTYPE_DECLARE_MACROS_H__


#define NT_GET_X_FLOATING_DTYPES_\
    X(::nt::float16_t, Float16, Half)\
    X(float, Float32, Float)\
    X(double, Float64, Double)\
    X(::nt::float128_t, Float128, LongDouble)\

#define NT_GET_X_COMPLEX_DTYPES_\
    X(::nt::complex_32, Complex32, cfloat16)\
    X(::nt::complex_64, Complex64, cfloat)\
    X(::nt::complex_128, Complex128, cdouble)


#define NT_GET_X_SIGNED_INTEGER_DTYPES_\
    X(int8_t, int8, Char)\
    X(int16_t, int16, Short)\
    X(int32_t, int32, Integer)\
    X(int64_t, int64, Long)\
    X(::nt::int128_t, int128, LongLong)\

#define NT_GET_X_UNSIGNED_INTEGER_DTYPES_\
    X(uint8_t, uint8, Byte)\
    X(uint16_t, uint16, UnsignedShort)\
    X(uint32_t, uint32, UnsignedInteger)\
    X(::nt::uint128_t, uint128, UnsignedLongLong)\


#define NT_GET_X_OTHER_DTYPES_\
    X(::nt::uint_bool_t, Bool, boolean)\
    X(::nt::Tensor, TensorObj, TensorData)\
    

#define NT_GET_DEFINE_FLOATING_DTYPES_(NT_CUR_FUNC__)\
    NT_CUR_FUNC__(::nt::float16_t, Float16, Half)\
    NT_CUR_FUNC__(float, Float32, Float)\
    NT_CUR_FUNC__(double, Float64, Double)\
    NT_CUR_FUNC__(::nt::float128_t, Float128, LongDouble)\

#define NT_GET_DEFINE_COMPLEX_DTYPES_(NT_CUR_FUNC__)\
    NT_CUR_FUNC__(::nt::complex_32, Complex32, cfloat16)\
    NT_CUR_FUNC__(::nt::complex_64, Complex64, cfloat)\
    NT_CUR_FUNC__(::nt::complex_128, Complex128, cdouble)


#define NT_GET_DEFINE_SIGNED_INTEGER_DTYPES_(NT_CUR_FUNC__)\
    NT_CUR_FUNC__(int8_t, int8, Char)\
    NT_CUR_FUNC__(int16_t, int16, Short)\
    NT_CUR_FUNC__(int32_t, int32, Integer)\
    NT_CUR_FUNC__(int64_t, int64, Long)\
    NT_CUR_FUNC__(::nt::int128_t, int128, LongLong)\

#define NT_GET_DEFINE_UNSIGNED_INTEGER_DTYPES_(NT_CUR_FUNC__)\
    NT_CUR_FUNC__(uint8_t, uint8, Byte)\
    NT_CUR_FUNC__(uint16_t, uint16, UnsignedShort)\
    NT_CUR_FUNC__(uint32_t, uint32, UnsignedInteger)\
    NT_CUR_FUNC__(::nt::uint128_t, uint128, UnsignedLongLong)\


#define NT_GET_DEFINE_OTHER_DTYPES_(NT_CUR_FUNC__)\
    NT_CUR_FUNC__(::nt::uint_bool_t, Bool, boolean)\
    NT_CUR_FUNC__(::nt::Tensor, TensorObj, TensorData)\

#endif
