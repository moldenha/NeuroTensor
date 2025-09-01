#ifndef NT_LAYERS_ADAPTIVE_AVG_POOL_2D_H_
#define NT_LAYERS_ADAPTIVE_AVG_POOL_2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API AdaptiveAvgPool2D : public Module {
  public:
    utils::my_tuple output_size;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveAvgPool2D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(output_size),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_tuple));
    // AdaptiveAvgPool2D(utils::my_tuple output_size);
    // template<typename T>
    // AdaptiveAvgPool2D(decltype(ntarg_(output_size))::equal_op_type<T> output_size)
    // :output_size(0)
    // {
    //     static_assert(std::is_convertible_v<T, int64_t>
    //                 || std::is_convertible_v<T, utils::my_tuple>
    //     , "Expected to get an integer value or 2D tuple for output size of AdaptiveAvgPool2D");
    //     if constexpr (std::is_convertible_v<T, int64_t>){
    //         output_size = utils::my_tuple(static_cast<int64_t>(output_size.val));
    //     }else{
    //         output_size = utils::my_tuple(output_size.val);
    //     }
    // }

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
