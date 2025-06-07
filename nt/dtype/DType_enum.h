#ifndef NT_DTYPE_ENUM_H
#define NT_DTYPE_ENUM_H
#include "../types/Types.h"

namespace nt{
enum class DType{
	Float,
	Float32 = Float,
	Double,
	Float64 = Double,
#ifdef _HALF_FLOAT_SUPPORT_
	Float16,
	Complex32,
	cfloat16=Complex32,
#endif
#ifdef _128_FLOAT_SUPPORT_
	Float128,
#endif
#ifdef __SIZEOF_INT128__
	int128,
	uint128,
#endif
	Complex64,
	cfloat = Complex64,
	Complex128,
	cdouble = Complex128,
	uint8,
	Byte = uint8,
	int8,
	Char = int8,
	int16,
	Short = int16,
	uint16,
	UnsignedShort = uint16,
	Integer,
	int32 = Integer,
	Long,
	uint32 = Long,
	int64,
	LongLong = int64,
	Bool,
	TensorObj
};

enum class DTypeShared{
	Float,
	Float32 = Float,
	Double,
	Float64 = Double,
#ifdef _HALF_FLOAT_SUPPORT_
	Float16,
	Complex32,
	cfloat16=Complex32,
#endif
#ifdef _128_FLOAT_SUPPORT_
	Float128,
#endif
#ifdef __SIZEOF_INT128__
	int128,
	uint128,
#endif
	Complex64,
	cfloat = Complex64,
	Complex128,
	cdouble = Complex128,
	uint8,
	Byte = uint8,
	int8,
	Char = int8,
	int16,
	Short = int16,
	uint16,
	UnsignedShort = uint16,
	Integer,
	int32 = Integer,
	Long,
	uint32 = Long,
	int64,
	LongLong = int64,
	Bool,
	TensorObj
};

inline DType DTypeShared_DType(const DTypeShared& dt){
	switch(dt){
		case DTypeShared::Integer:
			return DType::Integer;
#ifdef __SIZEOF_INT128__
		case DTypeShared::int128:
			return DType::int128;
		case DTypeShared::uint128:
			return DType::uint128;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DTypeShared::Float16:
			return DType::Float16;
		case DTypeShared::Complex32:
			return DType::Complex32;
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DTypeShared::Float128:
			return DType::Float128;
#endif
		case DTypeShared::Double:
			return DType::Double;
		case DTypeShared::Float:
			return DType::Float;
		case DTypeShared::Long:
			return DType::Long;
		case DTypeShared::cfloat:
			return DType::cfloat;
		case DTypeShared::cdouble:
			return DType::cdouble;
		case DTypeShared::uint8:
			return DType::uint8;
		case DTypeShared::int8:
			return DType::int8;
		case DTypeShared::int16:
			return DType::int16;
		case DTypeShared::uint16:
			return DType::uint16;
		case DTypeShared::int64:
			return DType::int64;
		case DTypeShared::TensorObj:
			return DType::TensorObj;
		case DTypeShared::Bool:
			return DType::Bool;
        default:
            return DType::Bool;
	}
}


inline DTypeShared DType_DTypeShared(const DType& dt){
	switch(dt){
		case DType::Integer:
			return DTypeShared::Integer;
#ifdef __SIZEOF_INT128__
		case DType::int128:
			return DTypeShared::int128;
		case DType::uint128:
			return DTypeShared::uint128;
#endif
#ifdef _HALF_FLOAT_SUPPORT_
		case DType::Float16:
			return DTypeShared::Float16;
		case DType::Complex32:
			return DTypeShared::Complex32;
#endif
#ifdef _128_FLOAT_SUPPORT_
		case DType::Float128:
			return DTypeShared::Float128;
#endif
		case DType::Double:
			return DTypeShared::Double;
		case DType::Float:
			return DTypeShared::Float;
		case DType::Long:
			return DTypeShared::Long;
		case DType::cfloat:
			return DTypeShared::cfloat;
		case DType::cdouble:
			return DTypeShared::cdouble;
		case DType::uint8:
			return DTypeShared::uint8;
		case DType::int8:
			return DTypeShared::int8;
		case DType::int16:
			return DTypeShared::int16;
		case DType::uint16:
			return DTypeShared::uint16;
		case DType::int64:
			return DTypeShared::int64;
		case DType::TensorObj:
			return DTypeShared::TensorObj;
		case DType::Bool:
			return DTypeShared::Bool;
        default:
            return DTypeShared::Bool;
	}

}

enum class DTypeConst{
	Float,
	Float32 = Float,
	Double,
	Float64 = Double,
	Complex64,
	cfloat = Complex64,
	Complex128,
	cdouble = Complex128,
	uint8,
	Byte = uint8,
	int8,
	Char = int8,
	int16,
	Short = int16,
	uint16,
	UnsignedShort = uint16,
	Integer,
	int32 = Integer,
	Long,
	uint32 = Long,
	int64,
	LongLong = int64,
	Bool,
	TensorObj,
	ConstFloat,
	ConstFloat32 = ConstFloat,
	ConstDouble,
	ConstFloat64 = ConstDouble,
	ConstComplex64,
	ConstCfloat = ConstComplex64,
	ConstComplex128,
	ConstCdouble = ConstComplex128,
	ConstUint8,
	ConstByte = ConstUint8,
	ConstInt8,
	ConstChar = ConstInt8,
	ConstInt16,
	ConstShort = ConstInt16,
	ConstUint16,
	ConstUnsignedShort = ConstUint16,
	ConstInteger,
	ConstInt32 = ConstInteger,
	ConstLong,
	ConstUint32 = ConstLong,
	ConstInt64,
	ConstLongLong = ConstInt64,
	ConstBool,
	ConstTensorObj,
#ifdef _HALF_FLOAT_SUPPORT_
	Float16,
	ConstFloat16,
	Complex32,
	ConstComplex32,
	cfloat16=Complex32,
	ConstCfloat16=ConstComplex32,
#endif
#ifdef _128_FLOAT_SUPPORT_
	Float128,
	ConstFloat128,
#endif
#ifdef __SIZEOF_INT128__
	int128,
	uint128,
	ConstInt128,
	ConstUint128
#endif
	
};


}

#endif
