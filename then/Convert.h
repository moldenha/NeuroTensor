#ifndef _NT_CONVERT_DTYPE_H_
#define _NT_CONVERT_DTYPE_H_

#include "std_convert.h"
#include "../Tensor.h"

namespace nt{
namespace convert{
template<DType T, typename A, std::enable_if_t<T == DType::TensorObj, bool> = true >
Tensor convert(const A&);

}
}

#endif // _NT_CONVERT_DTYPE_H_
