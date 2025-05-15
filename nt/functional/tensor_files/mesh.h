#ifndef __NT_FUNCTIONAL_TENSOR_FILES_MESH_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_MESH_H__

#include "../../Tensor.h"

namespace nt{
namespace functional{

Tensor one_hot(Tensor t, int64_t num_classes = -1);

}
}

#endif
