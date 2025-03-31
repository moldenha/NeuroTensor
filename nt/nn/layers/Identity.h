#ifndef _NT_LAYERS_IDENTITY_H_
#define _NT_LAYERS_IDENTITY_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class Identity : public Module {
  public:
    Identity() = default;
};

} // namespace layers
} // namespace nt

#endif
