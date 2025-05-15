#ifndef __NT_FUNCTIONAL_TENSOR_FILES_REPEAT_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_REPEAT_H__
#include "../../Tensor.h"

namespace nt{
namespace functional{

Tensor repeat_(const Tensor&, Tensor::size_value_t amt);
Tensor repeat_(const Tensor&, Tensor::size_value_t dim, Tensor::size_value_t amt);
Tensor expand(const Tensor&, SizeRef);
Tensor expand_as(const Tensor&, const Tensor&);

}
}

#endif
