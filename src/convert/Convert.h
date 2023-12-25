#ifndef _MY_CONVERT_DTYPE_H
#define _MY_CONVERT_DTYPE_H

#include "std_convert.h"
#include "../Tensor.h"

namespace nt{
namespace convert{
template<DType T, typename A, std::enable_if_t<T == DType::TensorObj, bool> = true >
Tensor convert(const A&);

}
}

#endif
