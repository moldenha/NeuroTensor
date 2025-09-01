#ifndef NT_FUNCTIONAL_TENSOR_FILES_DILATE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_DILATE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor undilate_(const Tensor&, Tensor::size_value_t);
NEUROTENSOR_API Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);

NEUROTENSOR_API Tensor undilate(const Tensor&, Tensor::size_value_t);
NEUROTENSOR_API Tensor undilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor undilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);


NEUROTENSOR_API Tensor dilate(const Tensor&, Tensor::size_value_t);
NEUROTENSOR_API Tensor dilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor dilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);


}
}

#endif
