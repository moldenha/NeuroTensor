#include "Functional.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

#define _NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(name, function)                    \
    namespace nt {                                                             \
    namespace layers {                                                         \
    name::name() { ; }                                                         \
    TensorGrad name::forward(TensorGrad x) { return functional::function(x); } \
    }                                                                          \
    }                                                                          \
    _NT_REGISTER_LAYER_NAMESPACED_(nt::layers::name, nt__layers__##name)

_NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(ReLU, relu)
_NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(SiLU, silu)
_NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(GELU, gelu)
_NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(Sigmoid, sigmoid)
_NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(Tanh, tanh)
_NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(Tan, tan)
_NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_(Log, log)

#undef _NT_DEFINE_FUNCTIONAL_SINGLE_LAYER_ 

