#include "ConvTranspose2D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

ConvTranspose2D::ConvTranspose2D(int64_t in_channels, int64_t out_channels, utils::my_tuple kernel_size,
         utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding,
         utils::my_tuple dilation, int64_t groups, bool use_bias)
      : use_bias(use_bias), groups(groups), in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding),
        output_padding(output_padding), dilation(dilation),
        Weight(functional::randn({in_channels, out_channels / groups,
                                  kernel_size[0], kernel_size[1]})),
        Bias(use_bias ? functional::randn({out_channels, 1, 1})
                      : Tensor::Null()) 
{
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    if(use_bias){
        this->register_parameter("Bias", Bias);
        Bias.detach().set_mutability(false);
    }
    Weight.detach().set_mutability(false);
}



TensorGrad ConvTranspose2D::forward(TensorGrad x) {
    utils::THROW_EXCEPTION(
        x.shape()[-3] == in_channels,
        "Expected input tensor to have channel size of $ but got $",
        in_channels, x.shape());
    TensorGrad outp =
        functional::conv_transpose2d(x, Weight, stride, padding, output_padding, dilation, groups);
    if (!use_bias) {
      return outp;
    }
    return outp + Bias;
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::ConvTranspose2D, nt__layers__ConvTranspose2D, use_bias,
                               groups, in_channels, out_channels, Weight)
