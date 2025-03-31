#ifndef _NT_LAYERS_FUNCTION_MIMIC_LAYERS_H_
#define _NT_LAYERS_FUNCTION_MIMIC_LAYERS_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"


#define _NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(name, function)                      \
  namespace nt {                                                               \
  namespace layers {                                                           \
  class name : public Module {                                                 \
  public:                                                                      \
    name();                                                                    \
    TensorGrad forward(TensorGrad x);                                          \
  };                                                                           \
  }                                                                            \
  }

_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(ReLU, relu)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(SiLU, silu)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(GELU, gelu)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Sigmoid, sigmoid)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Tanh, tanh)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Tan, tan)
_NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Log, log)

#undef _NT_MAKE_FUNCTIONAL_SINGLE_LAYER_ 

#endif
