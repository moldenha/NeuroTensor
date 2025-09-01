#ifndef NT_LAYERS_IDENTITY_H_
#define NT_LAYERS_IDENTITY_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class NEUROTENSOR_API Identity : public Module {
  public:
    Identity() = default;
};

} // namespace layers
} // namespace nt

#endif
