#ifndef __NT_FUNCTIONAL_TENSOR_FILES_DILATE_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_DILATE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

Tensor undilate_(const Tensor&, Tensor::size_value_t);
Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);

Tensor dilate(const Tensor&, Tensor::size_value_t);
Tensor dilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
Tensor dilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);


}
}

#endif
