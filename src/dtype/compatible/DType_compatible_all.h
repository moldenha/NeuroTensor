#ifndef __DTYPE_COMPATIBLE_ALL_H__
#define __DTYPE_COMPATIBLE_ALL_H__
#include "../../types/Types.h"
#if defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)
#include "../DType.h"
#include "../DType_enum.h"

namespace nt{

using ComplexTypesL = DTypeEnum<DType::Complex128, DType::Complex64, DType::Complex32>;
using FloatingTypesL = DTypeEnum<DType::Float128, DType::Float64, DType::Float32, DType::Float16>;
using IntegerTypesL = DTypeEnum<DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128>;
using SignedTypesL = DTypeEnum<DType::int32, DType::int64, DType::int16,DType::int8, DType::int128>;
using UnsignedTypesL = DTypeEnum<DType::uint32, DType::uint16, DType::uint8, DType::uint128>;
using AllTypesL = DTypeEnum<DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128, DType::Float16, DType::Complex128, DType::Complex64, DType::Complex32, DType::Bool, DType::TensorObj>;
using AllTypesNBoolL = DTypeEnum<DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128, DType::Float16, DType::Complex128, DType::Complex64, DType::Complex32, DType::TensorObj>;
using NumberTypesL = DTypeEnum<DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128, DType::Float16, DType::Complex128, DType::Complex64, DType::Complex32>; 
using RealNumberTypesL = DTypeEnum<DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128, DType::Float16>; 

namespace DTypeFuncs{
template<DType dt>
inline constexpr bool is_dtype_integer_v = is_in_t<dt, DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::int8>::value;
template<DType dt>
inline constexpr bool is_dtype_floating_v = is_in_v<dt, DType::Float, DType::Double, DType::Float128, DType::Float16>;
template<DType dt>
inline constexpr bool is_dtype_complex_v = is_in_v<dt, DType::Complex128, DType::Complex64, DType::Complex32>;
template<DType dt>
inline constexpr bool is_dtype_other_v = is_in_v<dt, DType::Bool, DType::TensorObj>;

template<DType dt>
inline constexpr bool is_dtype_real_num_v = (is_dtype_integer_v<dt> || is_dtype_floating_v<dt>);

template<DType dt>
inline constexpr bool is_dtype_num_v = (is_dtype_integer_v<dt> || is_dtype_floating_v<dt> || is_dtype_complex_v<dt>);


template<DType... dts>
inline constexpr bool integer_is_in_dtype_v = (is_in_v<DType::Integer, dts...>
					|| is_in_v<DType::Byte, dts...>
					|| is_in_v<DType::Short, dts...>
					|| is_in_v<DType::UnsignedShort, dts...>
					|| is_in_v<DType::LongLong, dts...>
					|| is_in_v<DType::Long, dts...>
					|| is_in_v<DType::Char, dts...>
					|| is_in_v<DType::int128, dts...>
					|| is_in_v<DType::uint128, dts...>);

template<DType... Rest>
struct dtype_to_type{
	using type = uint_bool_t;
};

template<DType dt, DType... Rest>
struct dtype_to_type<dt, Rest...>{
	using type = std::conditional_t<is_in_v<DType::Integer, dt, Rest...>, int32_t,
			std::conditional_t<is_in_v<DType::int128, dt, Rest...>, int128_t,
			std::conditional_t<is_in_v<DType::uint128, dt, Rest...>, uint128_t,
			std::conditional_t<is_in_v<DType::Float16, dt, Rest...>, float16_t,
			std::conditional_t<is_in_v<DType::Float128, dt, Rest...>, float128_t,
			std::conditional_t<is_in_v<DType::Complex32, dt, Rest...>, complex_32,
			std::conditional_t<is_in_v<DType::Double, dt, Rest...>, double,
			std::conditional_t<is_in_v<DType::Float, dt, Rest...>, float,
			std::conditional_t<is_in_v<DType::Long, dt, Rest...>, uint32_t,
			std::conditional_t<is_in_v<DType::TensorObj, dt, Rest...>, Tensor,
			std::conditional_t<is_in_v<DType::cfloat, dt, Rest...>, complex_64,
			std::conditional_t<is_in_v<DType::cdouble, dt, Rest...>, complex_128,
			std::conditional_t<is_in_v<DType::uint8, dt, Rest...>, uint8_t,
			std::conditional_t<is_in_v<DType::int8, dt, Rest...>, int8_t,
			std::conditional_t<is_in_v<DType::int16, dt, Rest...>, int16_t,
			std::conditional_t<is_in_v<DType::uint16, dt, Rest...>, uint16_t,
			std::conditional_t<is_in_v<DType::int64, dt, Rest...>, int64_t, uint_bool_t> > > > > > > > > > > > > > > > >;
};

template<DType dt>
struct dtype_to_type<dt>{
	using type = std::conditional_t<dt == DType::Integer, int32_t,
			std::conditional_t<dt == DType::int128, int128_t,
			std::conditional_t<dt == DType::uint128, uint128_t,
			std::conditional_t<dt == DType::Float16, float16_t,
			std::conditional_t<dt == DType::Float128, float128_t,
			std::conditional_t<dt == DType::Complex32, complex_32,
			std::conditional_t<dt == DType::Double, double,
			std::conditional_t<dt == DType::Float, float,
			std::conditional_t<dt == DType::Long, uint32_t,
			std::conditional_t<dt == DType::TensorObj, Tensor,
			std::conditional_t<dt == DType::cfloat, complex_64,
			std::conditional_t<dt == DType::cdouble, complex_128,
			std::conditional_t<dt == DType::uint8, uint8_t,
			std::conditional_t<dt == DType::int8, int8_t,
			std::conditional_t<dt == DType::int16, int16_t,
			std::conditional_t<dt == DType::uint16, uint16_t,
			std::conditional_t<dt == DType::int64, int64_t, uint_bool_t> > > > > > > > > > > > > > > > >;
};



template<DType... Rest>
using dtype_to_type_t = typename dtype_to_type<Rest...>::type; 


/* template<DType dt, DType... Rest> */
/* using dtype_to_type_t = typename dtype_to_type_multi2<dt, Rest...>::type; */

/* template<DType dt> */
/* using dtype_to_type_t = typename dtype_to_type_single<dt>::type; */


template<DType dt>
constexpr std::size_t size_of_dtype_c = sizeof(dtype_to_type_t<dt>);

template<DType dt>
constexpr bool is_convertible_to_complex = (size_of_dtype_c<dt> == 16 || size_of_dtype_c<dt> == 8 || size_of_dtype_c<dt> == 4);
template<DType dt>
inline constexpr DType convert_to_complex = size_of_dtype_c<dt> == 16 ? DType::Complex128 : size_of_dtype_c<dt> == 8 ? DType::Complex64 : size_of_dtype_c<dt> == 4 ? DType::Complex32 : DType::Bool;

template<DType dt>
constexpr bool is_convertible_to_floating = (size_of_dtype_c<dt> == 16 || size_of_dtype_c<dt> == 8 || size_of_dtype_c<dt> == 4 || size_of_dtype_c<dt> == 2);
template<DType dt>
inline constexpr DType convert_to_floating = size_of_dtype_c<dt> == 16 ? DType::Float128 : size_of_dtype_c<dt> == 8 ? DType::Float64 : size_of_dtype_c<dt> == 4 ? DType::Float32 : size_of_dtype_c<dt> == 2 ? DType::Float16 : DType::Bool;

template<DType dt>
constexpr bool is_convertible_to_integer = (size_of_dtype_c<dt> == 16 || size_of_dtype_c<dt> == 8 || size_of_dtype_c<dt> == 4 || size_of_dtype_c<dt> == 2 || size_of_dtype_c<dt> == 1) && dt != DType::Bool;
template<DType dt>
inline constexpr DType convert_to_integer = size_of_dtype_c<dt> == 16 ? DType::int128 : size_of_dtype_c<dt> == 8 ? DType::int64 : size_of_dtype_c<dt> == 4 ? DType::int32 : size_of_dtype_c<dt> == 2 ? DType::int16 : size_of_dtype_c<dt> == 1 ? DType::int8 : DType::Bool;


template<DType dt>
constexpr bool is_convertible_to_unsigned = is_convertible_to_integer<dt>;
template<DType dt>
inline constexpr DType convert_to_unsigned = size_of_dtype_c<dt> == 16 ? DType::uint128 : size_of_dtype_c<dt> == 4 ? DType::uint32 : size_of_dtype_c<dt> == 2 ? DType::uint16 : size_of_dtype_c<dt> == 1 ? DType::uint8 : DType::Bool;

template<typename T>
inline static constexpr DType type_to_dtype = (std::is_same_v<T, int32_t> ? DType::Integer
					: std::is_same_v<T, int128_t> ? DType::int128
					: std::is_same_v<T, uint128_t> ? DType::uint128
					: std::is_same_v<T, float16_t> ? DType::Float16
					: std::is_same_v<T, float128_t> ? DType::Float128
					: std::is_same_v<T, complex_32 > ? DType::Complex32
					: std::is_same_v<T, double> ? DType::Double
					: std::is_same_v<T, float> ? DType::Float
					: std::is_same_v<T, uint32_t> ? DType::Long
					: std::is_same_v<T, complex_64> ? DType::cfloat
					: std::is_same_v<T, complex_128> ? DType::cdouble
					: std::is_same_v<T, uint8_t> ? DType::uint8
					: std::is_same_v<T, int8_t> ? DType::int8
					: std::is_same_v<T, int16_t> ? DType::int16
					: std::is_same_v<T, int64_t> ? DType::int64
					: std::is_same_v<T, Tensor> ? DType::TensorObj
					: std::is_same_v<T, uint16_t> ? DType::uint16 : DType::Bool);


template <DType dt>
inline static constexpr bool dtype_is_num = is_in_v<dt, DType::uint128, DType::int128, DType::Float16, DType::Float128, DType::Complex32, DType::Integer, DType::Long, DType::uint8, DType::int8, DType::int16, DType::int64, DType::Float, DType::Double, DType::Complex128, DType::Complex64>;


//this is an iterator of dtypes (think like the next dtype to be checked, implemented below)
template<DType dt>
constexpr DType next_dtype_it = (dt == DType::Bool ? DType::Integer
					: dt == DType::Integer ? DType::int128
					: dt == DType::int128 ? DType::uint128
					: dt == DType::uint128 ? DType::Float16
					: dt == DType::Float16 ? DType::Float128
					: dt == DType::Float128 ? DType::Complex32
					: dt == DType::Complex32 ? DType::Double
					: dt == DType::Double ? DType::Float
					: dt == DType::Float ? DType::Long
					: dt == DType::Long ? DType::cfloat
					: dt == DType::cfloat ? DType::cdouble
					: dt == DType::cdouble ? DType::uint8
					: dt == DType::uint8 ? DType::int8
					: dt == DType::int8 ? DType::int16
					: dt == DType::int16 ? DType::int64
					: dt == DType::int64 ? DType::TensorObj
					: dt == DType::TensorObj ? DType::uint16 : DType::Bool);




 

}
}

#endif
#endif
