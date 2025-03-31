#ifndef __DTYPE_COMPATIBLE_FLOAT128_INT128_H__
#define __DTYPE_COMPATIBLE_FLOAT128_INT128_H__
#include "../../types/Types.h"
#if !defined(_HALF_FLOAT_SUPPORT_) && defined(_128_FLOAT_SUPPORT_) && defined(__SIZEOF_INT128__)
#include "../DType.h"
#include "../DType_enum.h"
#include "../Convert.h"

namespace nt{
using ComplexTypesL = DTypeEnum<DType::Complex128, DType::Complex64>;
using FloatingTypesL = DTypeEnum<DType::Float128, DType::Float64, DType::Float32>;
using IntegerTypesL = DTypeEnum<DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128>;
using SignedTypesL = DTypeEnum<DType::int32, DType::int64, DType::int16,DType::int8, DType::int128>;
using UnsignedTypesL = DTypeEnum<DType::uint32, DType::uint16, DType::uint8, DType::uint128>;
using AllTypesL = DTypeEnum<DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128, DType::Complex128, DType::Complex64, DType::Bool, DType::TensorObj>;
using AllTypesNBoolL = DTypeEnum<Type::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128, DType::Complex128, DType::Complex64, DType::TensorObj>;
using NumberTypesL = DTypeEnum<Type::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128, DType::Complex128, DType::Complex64>; 
using RealNumberTypesL = DTypeEnum<Type::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128, DType::Float, DType::Double, DType::Float128>; 

namespace DTypeFuncs{
template<DType dt>
inline constexpr bool is_dtype_integer_v = is_in_v<dt, DType::Integer, DType::Byte, DType::Short, DType::UnsignedShort, DType::LongLong, DType::Long, DType::int128, DType::uint128>;
template<DType dt>
inline constexpr bool is_dtype_floating_v = is_in_v<dt, DType::Float, DType::Double, DType::Float128>;
template<DType dt>
inline constexpr bool is_dtype_complex_v = is_in_v<dt, DType::Complex128, DType::Complex64>;
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
	using type = std::conditional_t<is_in_v<DType::Integer, Rest...>, int32_t,
			std::conditional_t<is_in_v<DType::int128, Rest...>, int128_t,
			std::conditional_t<is_in_v<DType::uint128, Rest...>, uint128_t,
			std::conditional_t<is_in_v<DType::Float128, Rest...>, float128_t,
			std::conditional_t<is_in_v<DType::Double, Rest...>, double,
			std::conditional_t<is_in_v<DType::Float, Rest...>, float,
			std::conditional_t<is_in_v<DType::Long, Rest...>, uint32_t,
			std::conditional_t<is_in_v<DType::TensorObj, Rest...>, Tensor,
			std::conditional_t<is_in_v<DType::cfloat, Rest...>, complex_64,
			std::conditional_t<is_in_v<DType::cdouble, Rest...>, complex_128,
			std::conditional_t<is_in_v<DType::uint8, Rest...>, uint8_t,
			std::conditional_t<is_in_v<DType::int8, Rest...>, int8_t,
			std::conditional_t<is_in_v<DType::int16, Rest...>, int16_t,
			std::conditional_t<is_in_v<DType::uint16, Rest...>, uint16_t,
			std::conditional_t<is_in_v<DType::int64, Rest...>, int64_t, uint_bool_t> > > > > > > > > > > > > > >;
};

template<DType dt>
struct dtype_to_type<dt>{
	using type = std::conditional_t<dt == DType::Integer, int32_t,
			std::conditional_t<dt == DType::int128, int128_t,
			std::conditional_t<dt == DType::uint128, uint128_t,
			std::conditional_t<dt == DType::Float128, float128_t,
			std::conditional_t<dt == DType::Double, double,
			std::conditional_t<dt == DType::Float, float,
			std::conditional_t<dt == DType::Long, uint32_t,
			std::conditional_t<dt == DType::TensorObj, Tensor,
			std::conditional_t<dt == DType::cfloat, complex_64,
			std::conditional_t<dt == DType::cdouble, complex_128,
			std::conditional_t<dt == DType::uint8, uint8_t,
			std::conditional_t<dt == DType::int8, int8_t,
			std::conditional_t<dt == DType::int16, int16_t,
			std::conditonal_t<dt == DType::uint16, uint16_t,
			std::conditional_t<dt == DType::int64, int64_t, uint_bool_t> > > > > > > > > > > > > > >;
};


template<DType... Rest>
using dtype_to_type_t = typename dtype_to_type<Rest...>::type;

template<DType dt>
using dtype_to_type_t = typename dtype_to_type<dt>::type;


template<DType dt>
constexpr std::size_t size_of_dtype = sizeof(dtype_to_type_t<dt>);

template<DType dt>
constexpr bool is_convertible_to_complex = (size_of_dtype<dt> == 16 || size_of_dtype<dt> == 8);
template<DType dt>
inline constexpr DType convert_to_complex = size_of_dtype<dt> == 16 ? DType::Complex128 : size_of_dtype<dt> == 8 ? DType::Complex64 : DType::Bool;

template<DType dt>
constexpr bool is_convertible_to_floating = (size_of_dtype<dt> == 16 || size_of_dtype<dt> == 8 || size_of_dtype<dt> == 4);
template<DType dt>
inline constexpr DType convert_to_floating = size_of_dtype<dt> == 16 ? DType::Float128 : size_of_dtype<dt> == 8 ? DType::Float64 : size_of_dtype<dt> == 4 ? DType::Float32 : DType::Bool;

template<DType dt>
constexpr bool is_convertible_to_integer = (size_of_dtype<dt> == 16 || size_of_dtype<dt> == 8 || size_of_dtype<dt> == 4 || size_of_dtype<dt> == 2 || size_of_dtype<dt> == 1) && dt != DType::Bool;
template<DType dt>
inline constexpr DType convert_to_integer = size_of_dtype<dt> == 16 ? DType::int128 : size_of_dtype<dt> == 8 ? DType::int64 : size_of_dtype<dt> == 4 ? DType::int32 : size_of_dtype<dt> == 2 ? DType::int16 : size_of_dtype<dt> == 1 ? DType::int8 : DType::Bool;


template<DType dt>
constexpr bool is_convertible_to_unsigned = is_convertible_to_integer<dt>;
template<DType dt>
inline constexpr DType convert_to_integer = size_of_dtype<dt> == 16 ? DType::uint128 : size_of_dtype<dt> == 8 ? DType::uint64 : size_of_dtype<dt> == 4 ? DType::uint32 : size_of_dtype<dt> == 2 ? DType::uint16 : size_of_dtype<dt> == 1 ? DType::uint8 : DType::Bool;

template<typename T>
inline static constexpr DType type_to_dtype = (std::is_same_v<T, int32_t> ? DType::Integer
					: std::is_same_v<T, int128_t> ? DType::int128
					: std::is_same_v<T, uint128_t> ? DType::uint128
					: std::is_same_v<T, float128_t> ? DType::Float128
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
inline static constexpr bool dtype_is_num = is_in_v<dt, DType::uint128, DType::int128, DType::Float128, DType::Integer, DType::Long, DType::uint8, DType::int8, DType::int16, DType::int64, DType::uint64, DType::Float, DType::Double, DType::Complex128, DType::Complex64>;

bool is_unsiged(const DType& dt){return ((dt == DType::Long) || (dt == DType::uint16) || (dt == DType::uint8) || (dt == DType::uint128));}
bool is_integer(const DType& dt){return is_in<DType::Integer, DType::Long, DType::uint8, DType::int8, DType::int16, DType::int64, DType::uint64, DType::int128, DType::uint128>(dt);}
bool is_floating(const DType& dt){return ((dt == DType::Float) || (dt == DType::Double) || (dt == DType::Float128));}
bool is_complex(const DType& dt){return ((dt == DType::Complex64 ||dt == DType::Complex128));

std::size_t size_of_dtype(const DType& dt){
	switch(dt){
		case DType::Integer:
			return sizeof(int32_t);
		case DType::int128:
			return sizeof(int128_t);
		case DType::uint128:
			return sizeof(uint128_t);
		case DType::Float128:
			return sizeof(float128_t);
		case DType::Double:
			return sizeof(double);
		case DType::Float:
			return sizeof(float);
		case DType::Long:
			return sizeof(uint32_t);
		case DType::cfloat:
			return sizeof(complex_64);
		case DType::cdouble:
			return sizeof(complex_128);
		case DType::uint8:
			return sizeof(uint8_t);
		case DType::int8:
			return sizeof(int8_t);
		case DType::int16:
			return sizeof(int16_t);
		case DType::uint16:
			return sizeof(uint16_t);
		case DType::int64:
			return sizeof(int64_t);
		case DType::TensorObj:
			return sizeof(Tensor);
		case DType::Bool:
			return sizeof(uint_bool_t);
	}
}



bool can_convert(const DType& from, const DType& to){
	return size_of_dtype(from) == size_of_dtype(to);
}

//this is an iterator of dtypes (think like the next dtype to be checked, implemented below)
template<DType dt>
constexpr DType next_dtype_it = (dt == DType::Bool ? DType::Integer
					: dt == DType::Integer ? DType::int128
					: dt == DType::int128 ? DType::uint128
					: dt == DType::uint128 ? DType::Float128
					: dt == DType::Float128 ? DType::Double
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
