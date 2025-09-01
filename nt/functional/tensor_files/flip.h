#ifndef NT_FUNCTIONAL_TENSOR_FILES_FLIP_H__
#define NT_FUNCTIONAL_TENSOR_FILES_FLIP_H__

#include "../../Tensor.h"
#include "../../utils/optional_list.h"


namespace nt {
namespace functional {

NEUROTENSOR_API Tensor flip_view(const Tensor&, utils::optional_list list = nullptr); 
NEUROTENSOR_API Tensor flip(const Tensor&, utils::optional_list list = nullptr);


} // namespace functional
} // namespace nt

#endif
