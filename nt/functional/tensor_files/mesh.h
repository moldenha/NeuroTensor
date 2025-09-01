#ifndef NT_FUNCTIONAL_TENSOR_FILES_MESH_H__
#define NT_FUNCTIONAL_TENSOR_FILES_MESH_H__

#include "../../Tensor.h"

namespace nt{
namespace functional{

NEUROTENSOR_API Tensor one_hot(const Tensor& t, int64_t num_classes = -1);
NEUROTENSOR_API Tensor meshgrid(const Tensor &, const Tensor &);

}
}

#endif
