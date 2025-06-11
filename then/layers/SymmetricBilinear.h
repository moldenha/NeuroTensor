#ifndef _NT_LAYERS_SYMMETRIC_BILINEAR_H_
#define _NT_LAYERS_SYMMETRIC_BILINEAR_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {

class SymmetricBilinear : public Module{
  public:
    bool use_bias;
    TensorGrad W1, W2, Bias;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(SymmetricBilinear,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(input_size, hidden_size, use_bias),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(true),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, bool));
    // SymmetricBilinear(int64_t input_size, int64_t hidden_size,
    //                   bool use_bias = true);
    TensorGrad forward(TensorGrad);
};

} // namespace layers
} // namespace nt

#endif
