#ifndef _NT_SPARSE_COMMA_OPERATOR_H_
#define _NT_SPARSE_COMMA_OPERATOR_H_

namespace nt{
class SparseCommaOperator;
}

#include "../Tensor.h"
#include <vector>

namespace nt {
namespace DTypeFuncs {
namespace detail_sparse_comma_operator {
// this is where functions that are going to be used to convert a scalar into an
// increment are going to be used
#define _NT_SPARSE_ADJUST_FUNC_(type dtype)\
template <typename Func>
// inline int64_t 
std::size_t setFloat(void*, Scalar);
std::size_t setDouble(void*, Scalar);
std::size_t setComplex64(void*, Scalar);
std::size_t setComplex128(void*, Scalar);
std::size_t setint8(void*, Scalar);
std::size_t setuint8(void*, Scalar);
std::size_t setuint16(void*, Scalar);
std::size_t setint16(void*, Scalar);
std::size_t setint32(void*, Scalar);
std::size_t setuint32(void*, Scalar);
std::size_t setint64(void*, Scalar);
std::size_t setBool(void*, Scalar);
std::size_t setFloat16(void*, Scalar);
std::size_t setComplex32(void*, Scalar);
#ifdef _128_FLOAT_SUPPORT_
std::size_t setFloat128(void*, Scalar);
#endif
#ifdef __SIZEOF_INT128__
std::size_t setint128(void*, Scalar);
std::size_t setuint128(void*, Scalar);
#endif

using SetFunc = std::size_t (*)(void *, Scalar);

inline SetFunc getSetFunc(DType dt){
    switch(dt){
        case DType::Float32:
            return setFloat;
        case DType::Float64:
            return setDouble;
        case DType::Complex64:
            return setComplex64;
        case DType::Complex128:
            return setComplex128;
        case DType::int8:
            return setint8;
        case DType::uint8:
            return setuint8;
        case DType::int16:
            return setint16;
        case DType::uint16:
            return setuint16;
        case DType::int32:
            return setint32;
        case DType::uint32:
            return setuint32;
        case DType::int64:
            return setint64;
        case DType::Bool:
            return setBool;
        case DType::Float16:
            return setFloat16;
        case DType::Complex32:
            return setComplex32;
#ifdef _128_FLOAT_SUPPORT_
        case DType::Float128:
            return setFloat128;
#endif
#ifdef __SIZEOF_INT128__
        case DType::int128:
            return setint128;
        case DType::uint128:
            return setuint128;
#endif
        default:
            utils::throw_exception(false, "unsupported dtype for comma operator $", dt);
            return setBool;
            break;
    }
}

}
} // namespace DTypeFuncs

// class CommaOperator{
//     DTypeFuncs::detail_comma_operator::SetFunc set_func;
//     void* begin;
//     void* end;
//     DType dt;
//     CommaOperator(DTypeFuncs::detail_comma_operator::SetFunc, void*, void*, DType);
// public:
//     CommaOperator() = delete;
//     CommaOperator(void*, void*, DType);
//     CommaOperator(const CommaOperator&);
//     CommaOperator(CommaOperator&&);
//     CommaOperator operator,(Scalar);
//     CommaOperator operator,(Tensor);
// };

} // namespace nt


#endif
