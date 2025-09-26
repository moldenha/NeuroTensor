#include "ConvTransposeND.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

inline SizeRef make_weight_size(int64_t dim, int64_t in_channels, int64_t out_channels, int64_t groups, utils::optional_list kernel_size){
    std::vector<int64_t> out_size(dim+2);
    out_size[0] = in_channels;
    out_size[1] = out_channels / groups;
    std::vector<int64_t> copying = kernel_size.to_repeat_vector(dim);
    std::copy(copying.cbegin(), copying.cend(), out_size.begin()+2);
    return SizeRef(std::move(out_size));
}

inline SizeRef make_bias_size(int64_t dim, int64_t out_channels){
    std::vector<int64_t> out_size(dim+1, 1);
    out_size[0] = out_channels;
    return SizeRef(std::move(out_size));
}

ConvTransposeND::ConvTransposeND(int64_t dim, int64_t in_channels, int64_t out_channels, utils::optional_list kernel_size,
           utils::optional_list stride, utils::optional_list padding, utils::optional_list output_padding,
           utils::optional_list dilation,
           int64_t groups, bool use_bias)
        :use_bias(use_bias), groups(groups), dim(dim) in_channels(in_channels),
        out_channels(out_channels), stride(stride), padding(padding), output_padding(output_padding),
        dilation(dilation),
        Weight(functional::randn(make_weight_size(dim, in_channels, out_channels, groups, kernel_size))),
        Bias(use_bias ? functional::randn(make_bias_size(dim, out_channels))
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



TensorGrad ConvTransposeND::forward(TensorGrad x) {
    utils::THROW_EXCEPTION(
        x.shape()[-1 * (this->dim+1)] == in_channels,
        "Expected input tensor to have channel size of $ but got $",
        in_channels, x.shape());
    TensorGrad outp = 
        functional::conv_transposend(x, Weight, dim, stride, padding, output_padding, dilation, groups);
    if(use_bias){return outp + Bias;}
    return std::move(outp);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::ConvTransposeND, nt__layers__ConvTransposeND, use_bias,
                               groups, dim, in_channels, out_channels, Weight)
