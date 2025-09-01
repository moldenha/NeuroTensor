#ifndef NT_FUNCTIONAL_TENSOR_FILES_TRANSPOSE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_TRANSPOSE_H__

#include "../../Tensor.h"
#include <vector>

namespace nt{
namespace functional{

NEUROTENSOR_API Tensor transpose(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor& row_col_swap_(Tensor&);
NEUROTENSOR_API Tensor permute(const Tensor&, std::vector<Tensor::size_value_t>);

}
}

#endif
