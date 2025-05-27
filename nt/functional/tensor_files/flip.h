#ifndef __NT_FUNCTIONAL_TENSOR_FILES_FLIP_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_FLIP_H__

#include "../../Tensor.h"
#include "../../utils/optional_list.h"


namespace nt {
namespace functional {

Tensor flip_view(const Tensor&, utils::optional_list list = nullptr); 
Tensor flip(const Tensor&, utils::optional_list list = nullptr);


} // namespace functional
} // namespace nt

#endif
