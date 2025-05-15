#include "Conv3D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Conv3D::Conv3D(int64_t in_channels, int64_t out_channels,
       utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride,
       utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation,
       int64_t groups, bool use_bias)
    : use_bias(use_bias), groups(groups), in_channels(in_channels),
      out_channels(out_channels), stride(stride), padding(padding),
      dilation(dilation),
      Weight(
          functional::randn({out_channels, in_channels / groups, kernel_size[0],
                             kernel_size[1], kernel_size[3]})),
      Bias(use_bias ? functional::randn({out_channels, 1, 1, 1})
                    : Tensor::Null()) {
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    if(use_bias){
        this->register_parameter("Bias", Bias);
        Bias.tensor.set_mutability(false);
    }
    Weight.tensor.set_mutability(false);
}

TensorGrad Conv3D::forward(TensorGrad x) {
    utils::THROW_EXCEPTION(
        x.shape()[-4] == in_channels,
        "Expected input tensor to have channel size of $ but got $",
        in_channels, x.shape());
    TensorGrad outp =
        functional::conv3d(x, Weight, stride, padding, dilation, groups);
    if (use_bias) {
        return outp + Bias;
    }
    return std::move(outp);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Conv3D, nt__layers__Conv3D, use_bias,
                               groups, in_channels, out_channels, Weight)
