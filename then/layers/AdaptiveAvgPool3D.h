#ifndef _NT_LAYERS_ADAPTIVE_AVG_POOL_3D_H_
#define _NT_LAYERS_ADAPTIVE_AVG_POOL_3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class AdaptiveAvgPool3D : public Module {
  public:
    utils::my_n_tuple<3> output_size;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveAvgPool3D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(output_size),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_n_tuple<3>));
    // AdaptiveAvgPool3D(utils::my_n_tuple<3> output_size);
    // template<typename T>
    // AdaptiveAvgPool3D(decltype(ntarg_(output_size))::equal_op_type<T> output_size)
    // :output_size(0)
    // {
    //     static_assert(std::is_convertible_v<T, int64_t>
    //                 || std::is_convertible_v<T, utils::my_n_tuple<3>>
    //     , "Expected to get an integer value or 3D tuple for output size of AdaptiveAvgPool3D");
    //     if constexpr (std::is_convertible_v<T, int64_t>){
    //         output_size = utils::my_n_tuple<3>(static_cast<int64_t>(output_size.val));
    //     }else{
    //         output_size = utils::my_n_tuple<3>(output_size.val);
    //     }
    // }

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
