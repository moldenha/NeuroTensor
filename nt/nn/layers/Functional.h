#ifndef NT_LAYERS_FUNCTION_MIMIC_LAYERS_H_
#define NT_LAYERS_FUNCTION_MIMIC_LAYERS_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"


#define NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(name, function)                      \
  namespace nt {                                                               \
  namespace layers {                                                           \
  class NEUROTENSOR_API name : public Module {                                                 \
  public:                                                                      \
    name();                                                                    \
    TensorGrad forward(TensorGrad x);                                          \
  };                                                                           \
  }                                                                            \
  }

NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(ReLU, relu)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(SiLU, silu)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(GELU, gelu)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Sigmoid, sigmoid)

NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Tanh, tanh)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Tan, tan)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Sinh, tanh)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Sin, sin)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Cosh, cosh)
NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Cos, cos)

NT_MAKE_FUNCTIONAL_SINGLE_LAYER_(Log, log)

#undef NT_MAKE_FUNCTIONAL_SINGLE_LAYER_ 

#endif
