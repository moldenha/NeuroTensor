#ifndef _DTYPE_ENUM_H
#define _DTYPE_ENUM_H
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
