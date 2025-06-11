// these are functions specifically designed for min-max type functions
// this includes clamp, relu, etc
#ifndef __NT_FUNCTIONAL_TENSOR_FILES_DROPOUT_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_DROPOUT_H__

#include "../../Tensor.h"

namespace nt {
namespace functional {

Tensor dropout(const Tensor &, double);
Tensor dropout2d(const Tensor &, double);
Tensor dropout3d(const Tensor &, double);

} // namespace functional
} // namespace nt


#endif
