// these are functions specifically designed for min-max type functions
// this includes clamp, relu, etc
#ifndef NT_FUNCTIONAL_TENSOR_FILES_DROPOUT_H__
#define NT_FUNCTIONAL_TENSOR_FILES_DROPOUT_H__

#include "../../Tensor.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor dropout(const Tensor &, double);
NEUROTENSOR_API Tensor dropout2d(const Tensor &, double);
NEUROTENSOR_API Tensor dropout3d(const Tensor &, double);

} // namespace functional
} // namespace nt


#endif
