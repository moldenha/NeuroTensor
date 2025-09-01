#ifndef NT_SPARSE_COMMA_OPERATOR_H__
#define NT_SPARSE_COMMA_OPERATOR_H__

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

#define X(type, dtype_enum_a, dtype_enum_b)\
NEUROTENSOR_API std::size_t set##dtype_enum_a(void*, Scalar);\

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_

NEUROTENSOR_API std::size_t setBool(void*, Scalar);

#undef X

using SetFunc = std::size_t (*)(void *, Scalar);

NT_ALWAYS_INLINE SetFunc getSetFunc(DType dt){
    switch(dt){
#define X(type, dtype_enum_a, dtype_enum_b)\
        case DType::dtype_enum_a: return set##dtype_enum_a;\

NT_GET_X_FLOATING_DTYPES_ 
NT_GET_X_COMPLEX_DTYPES_
NT_GET_X_SIGNED_INTEGER_DTYPES_
NT_GET_X_UNSIGNED_INTEGER_DTYPES_
        case DType::Bool:
            return setBool;
        default:
            utils::throw_exception(false, "unsupported dtype for comma operator $", dt);
            return setBool;
            break;
#undef X
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
