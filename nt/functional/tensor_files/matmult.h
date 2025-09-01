#ifndef NT_TENSOR_FILES_FUNCTIONAL_MATMULT_H__
#define NT_TENSOR_FILES_FUNCTIONAL_MATMULT_H__

#include "../../Tensor.h"
#include <cstddef>

namespace nt{
namespace functional{

NEUROTENSOR_API Tensor matmult(const Tensor&, const Tensor&, bool trans_a = false, bool trans_b = false);
NEUROTENSOR_API Tensor& matmult(const Tensor&, const Tensor&, Tensor&, bool trans_a = false, bool trans_b = false);
NEUROTENSOR_API Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool trans_input = false, bool trans_weight = false);
NEUROTENSOR_API Tensor matmult_cT(const Tensor&, const Tensor&);



}

}

#endif
