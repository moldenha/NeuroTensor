#ifndef __NT_FUNCTIONAL_TENSOR_FILES_RANGES_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_RANGES_H__

#include "../../Tensor.h"

namespace nt{
namespace functional{
Tensor get_range(const Tensor &t, const my_range &r, size_t idx);
Tensor op_range(Tensor t, my_range);
Tensor op_range(const Tensor& t, std::vector<my_range>);
}
}

#endif
