#ifndef _NT_LAYERS_ADAPTIVE_AVG_POOL_1D_H_
#define _NT_LAYERS_ADAPTIVE_AVG_POOL_1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class AdaptiveAvgPool1D : public Module {
  public:
    int64_t output_size;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveAvgPool1D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(output_size),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t));
    // AdaptiveAvgPool1D(int64_t output_size);
    // template<typename T>
    // AdaptiveAvgPool1D(decltype(ntarg_(output_size))::equal_op_type<T> output_size)
    // :output_size(0)
    // {
    //     static_assert(std::is_convertible_v<T, int64_t>, "Expected to get an integer value for output size of AdaptiveAvgPool1D");
    //     output_size = static_cast<int64_t>(output_size.val);
    // }

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
