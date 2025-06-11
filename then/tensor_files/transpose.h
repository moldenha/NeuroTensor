#ifndef __NT_FUNCTIONAL_TENSOR_FILES_TRANSPOSE_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_TRANSPOSE_H__

#include "../../Tensor.h"
#include <vector>

namespace nt{
namespace functional{

Tensor transpose(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
Tensor& row_col_swap_(Tensor&);
Tensor permute(const Tensor&, std::vector<Tensor::size_value_t>);

}
}

#endif
