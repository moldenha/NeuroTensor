#include "Conv1D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

Conv1D::Conv1D(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
           int64_t stride, int64_t padding, int64_t dilation,
           int64_t groups, bool use_bias)
        : use_bias(use_bias), groups(groups), in_channels(in_channels),
          out_channels(out_channels), stride(stride), padding(padding),
          dilation(dilation),
          Weight(functional::randn(
              {out_channels, in_channels / groups, kernel_size})),
          Bias(use_bias ? functional::randn({out_channels, 1})
                        : Tensor::Null()) 
{
    utils::THROW_EXCEPTION(
        out_channels % groups == 0,
        "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(
        in_channels % groups == 0,
        "Expected in channels to be divisible by groups");
    if(use_bias){
        this->register_parameter("Bias", Bias);
        Bias.tensor.set_mutability(false);
    }
    Weight.tensor.set_mutability(false);
}

TensorGrad Conv1D::forward(TensorGrad x) {
    utils::THROW_EXCEPTION(
        x.shape()[-2] == in_channels,
        "Expected input tensor to have channel size of $ but got $",
        in_channels, x.shape());
    TensorGrad outp =
        functional::conv1d(x, Weight, stride, padding, dilation, groups);
    if (use_bias) {
        return outp + Bias;
    }
    return std::move(outp);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Conv1D, nt__layers__Conv1D, use_bias,
                               groups, in_channels, out_channels, Weight)

