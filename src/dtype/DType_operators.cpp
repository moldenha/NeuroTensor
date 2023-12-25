#include "DType_operators.h"

namespace nt{
namespace DTypeFuncs{

template<DType dt>
Tensor& MultiplyThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a *= b;}
template<DType dt>
Tensor& MultiplyThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a *= b;}

template<DType dt>
Tensor& DivideThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a /= b;}
template<DType dt>
Tensor& DivideThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a /= b;}

template<DType dt>
Tensor& SubtractThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a -= b;}
template<DType dt>
Tensor& SubtractThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a -= b;}

template<DType dt>
Tensor& AddThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Scalar& b){return a += b;}
template<DType dt>
Tensor& AddThis<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(Tensor& a, const Tensor& b){return a += b;}

template<DType dt>
Tensor Multiply<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a * b;}
template<DType dt>
Tensor Multiply<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a * b;}

template<DType dt>
Tensor Divide<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a / b;}
template<DType dt>
Tensor Divide<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a / b;}

template<DType dt>
Tensor Subtract<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a - b;}
template<DType dt>
Tensor Subtract<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a - b;}

template<DType dt>
Tensor Add<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Scalar& b){return a + b;}
template<DType dt>
Tensor Add<dt, std::enable_if_t<dt == DType::TensorObj, bool>>::operator()(const Tensor& a, const Tensor& b){return a + b;}



/* class MultiplyThis<DType::Float>; */
/* class MultiplyThis<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class MultiplyThis<DType::Float16>; */
/* class MultiplyThis<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class MultiplyThis<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class MultiplyThis<DType::uint128>; */
/* class MultiplyThis<DType::int128>; */
/* #endif */
/* class MultiplyThis<DType::int64>; */
/* class MultiplyThis<DType::uint32>; */
/* class MultiplyThis<DType::int32>; */
/* class MultiplyThis<DType::uint16>; */
/* class MultiplyThis<DType::int16>; */
/* class MultiplyThis<DType::uint8>; */
/* class MultiplyThis<DType::int8>; */
/* class MultiplyThis<DType::Complex128>; */
/* class MultiplyThis<DType::Complex64>; */
/* class MultiplyThis<DType::TensorObj>; */

/* class DivideThis<DType::Float>; */
/* class DivideThis<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class DivideThis<DType::Float16>; */
/* class DivideThis<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class DivideThis<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class DivideThis<DType::uint128>; */
/* class DivideThis<DType::int128>; */
/* #endif */
/* class DivideThis<DType::int64>; */
/* class DivideThis<DType::uint32>; */
/* class DivideThis<DType::int32>; */
/* class DivideThis<DType::uint16>; */
/* class DivideThis<DType::int16>; */
/* class DivideThis<DType::uint8>; */
/* class DivideThis<DType::int8>; */
/* class DivideThis<DType::Complex128>; */
/* class DivideThis<DType::Complex64>; */
/* class DivideThis<DType::TensorObj>; */

/* class AddThis<DType::Float>; */
/* class AddThis<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class AddThis<DType::Float16>; */
/* class AddThis<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class AddThis<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class AddThis<DType::uint128>; */
/* class AddThis<DType::int128>; */
/* #endif */
/* class AddThis<DType::int64>; */
/* class AddThis<DType::uint32>; */
/* class AddThis<DType::int32>; */
/* class AddThis<DType::uint16>; */
/* class AddThis<DType::int16>; */
/* class AddThis<DType::uint8>; */
/* class AddThis<DType::int8>; */
/* class AddThis<DType::Complex128>; */
/* class AddThis<DType::Complex64>; */
/* class AddThis<DType::TensorObj>; */

/* class SubtractThis<DType::Float>; */
/* class SubtractThis<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class SubtractThis<DType::Float16>; */
/* class SubtractThis<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class SubtractThis<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class SubtractThis<DType::uint128>; */
/* class SubtractThis<DType::int128>; */
/* #endif */
/* class SubtractThis<DType::int64>; */
/* class SubtractThis<DType::uint32>; */
/* class SubtractThis<DType::int32>; */
/* class SubtractThis<DType::uint16>; */
/* class SubtractThis<DType::int16>; */
/* class SubtractThis<DType::uint8>; */
/* class SubtractThis<DType::int8>; */
/* class SubtractThis<DType::Complex128>; */
/* class SubtractThis<DType::Complex64>; */
/* class SubtractThis<DType::TensorObj>; */

/* class Multiply<DType::Float>; */
/* class Multiply<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class Multiply<DType::Float16>; */
/* class Multiply<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class Multiply<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class Multiply<DType::uint128>; */
/* class Multiply<DType::int128>; */
/* #endif */
/* class Multiply<DType::int64>; */
/* class Multiply<DType::uint32>; */
/* class Multiply<DType::int32>; */
/* class Multiply<DType::uint16>; */
/* class Multiply<DType::int16>; */
/* class Multiply<DType::uint8>; */
/* class Multiply<DType::int8>; */
/* class Multiply<DType::Complex128>; */
/* class Multiply<DType::Complex64>; */
/* class Multiply<DType::TensorObj>; */

/* class Divide<DType::Float>; */
/* class Divide<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class Divide<DType::Float16>; */
/* class Divide<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class Divide<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class Divide<DType::uint128>; */
/* class Divide<DType::int128>; */
/* #endif */
/* class Divide<DType::int64>; */
/* class Divide<DType::uint32>; */
/* class Divide<DType::int32>; */
/* class Divide<DType::uint16>; */
/* class Divide<DType::int16>; */
/* class Divide<DType::uint8>; */
/* class Divide<DType::int8>; */
/* class Divide<DType::Complex128>; */
/* class Divide<DType::Complex64>; */
/* class Divide<DType::TensorObj>; */

/* class Add<DType::Float>; */
/* class Add<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class Add<DType::Float16>; */
/* class Add<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class Add<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class Add<DType::uint128>; */
/* class Add<DType::int128>; */
/* #endif */
/* class Add<DType::int64>; */
/* class Add<DType::uint32>; */
/* class Add<DType::int32>; */
/* class Add<DType::uint16>; */
/* class Add<DType::int16>; */
/* class Add<DType::uint8>; */
/* class Add<DType::int8>; */
/* class Add<DType::Complex128>; */
/* class Add<DType::Complex64>; */
/* class Add<DType::TensorObj>; */



/* class Subtract<DType::Float>; */
/* class Subtract<DType::Double>; */
/* #ifdef _HALF_FLOAT_SUPPORT_ */
/* class Subtract<DType::Float16>; */
/* class Subtract<DType::Complex32>; */
/* #endif */
/* #ifdef _128_FLOAT_SUPPORT_ */
/* class Subtract<DType::Float128>; */
/* #endif */
/* #ifdef __SIZEOF_INT128__ */
/* class Subtract<DType::uint128>; */
/* class Subtract<DType::int128>; */
/* #endif */
/* class Subtract<DType::int64>; */
/* class Subtract<DType::uint32>; */
/* class Subtract<DType::int32>; */
/* class Subtract<DType::uint16>; */
/* class Subtract<DType::int16>; */
/* class Subtract<DType::uint8>; */
/* class Subtract<DType::int8>; */
/* class Subtract<DType::Complex128>; */
/* class Subtract<DType::Complex64>; */
/* class Subtract<DType::TensorObj>; */


}
}
