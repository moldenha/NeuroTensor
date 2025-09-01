#include "ConvND.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Tensor ConvND::MakeKernel(utils::optional_list kernel_size, int64_t in_channels, int64_t out_channels, int64_t groups, int64_t dim){
    utils::throw_exception(kernel_size.has_value() && (kernel_size.is_scalar() || kernel_size->size() == dim),
                       "Error, expected kernel_size to be a single value or equal to the number of dimensions for conv$d", dim);
    utils::THROW_EXCEPTION(out_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    utils::THROW_EXCEPTION(in_channels % groups == 0,
                           "Expected in channels to be divisible by groups");
    std::vector<int64_t> shape(dim+2);
    shape[0] = out_channels;
    shape[1] = in_channels / groups;
    if(kernel_size.is_scalar()){
        const int64_t& s = kernel_size.get_scalar();
        for(int64_t i = 0; i < dim; ++i)
            shape[i+2] = s;
    }else{
        auto begin = kernel_size.cbegin();
        for(int64_t i = 0; i < dim; ++i, ++begin)
            shape[i+2] = *begin;
    }
    return functional::randn(SizeRef(std::move(shape)));
}

Tensor ConvND::MakeBias(int64_t out_channels, int64_t dim, bool use_bias){
    if(!use_bias) return Tensor::Null();
    std::vector<int64_t> shape(dim+1, 1);
    shape[0] = out_channels;
    return functional::randn(SizeRef(std::move(shape)));
}

ConvND::ConvND(int64_t in_channels, int64_t out_channels, int64_t dim,
       utils::optional_list kernel_size, utils::optional_list stride,
       utils::optional_list padding, utils::optional_list dilation,
       int64_t groups, bool use_bias)
    : use_bias(use_bias), groups(groups), in_channels(in_channels),
      out_channels(out_channels), dim(dim), stride(stride), padding(padding),
      dilation(dilation),
      Weight(ConvND::MakeKernel(kernel_size, in_channels, out_channels, groups, dim)),
      Bias(ConvND::MakeBias(out_channels, dim, use_bias)) {
    if(use_bias){
        this->register_parameter("Bias", Bias);
        Bias.detach().set_mutability(false);
    }
    Weight.detach().set_mutability(false);
}

TensorGrad ConvND::forward(TensorGrad x) {
    utils::THROW_EXCEPTION(
        x.shape()[-1 * (this->dim+1)] == in_channels,
        "Expected input tensor to have channel size of $ but got $",
        in_channels, x.shape());
    TensorGrad outp =
        functional::convnd(x, Weight, dim, stride, padding, dilation, groups);
    if (use_bias) {
        return outp + Bias;
    }
    return std::move(outp);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::ConvND, nt__layers__ConvND, use_bias,
                               groups, in_channels, out_channels, dim, Weight)
