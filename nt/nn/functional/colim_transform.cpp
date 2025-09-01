#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"
#include <algorithm>

namespace nt{
namespace functional{

TensorGrad TensorGrad_Functional_Class::unfold1d(
        const TensorGrad &x, Tensor::size_value_t kernel_size,
        Tensor::size_value_t dilation, Tensor::size_value_t padding,
        Tensor::size_value_t stride, bool transpose_out) {
    TensorGrad result(::nt::functional::unfold1d(x.detach(), kernel_size, dilation, padding, stride,
                                                         transpose_out), x.track_grad());
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                Tensor::size_value_t output_size = parents[0]->detach().shape()[-1];
                ::nt::functional::unfold1d_backward(grad, parents[0]->grad(), output_size,
                                                    kernel_size, dilation, padding, stride,
                                                    transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::unfold2d(
        const TensorGrad &x, utils::my_tuple kernel_size, utils::my_tuple dilation,
        utils::my_tuple padding, utils::my_tuple stride, bool transpose_out) {
    TensorGrad result(
            ::nt::functional::unfold2d(x.detach(), kernel_size, dilation, padding, stride, transpose_out), x.track_grad());
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                utils::my_tuple output_size(parents[0]->detach().shape()[-2],
                                            parents[0]->detach().shape()[-1]);
                ::nt::functional::unfold2d_backward(grad, parents[0]->grad(), output_size,
                                                kernel_size, dilation, padding, stride, transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::unfold3d(
        const TensorGrad &x, utils::my_n_tuple<3> kernel_size,
        utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding,
        utils::my_n_tuple<3> stride, bool transpose_out) {
    TensorGrad result(::nt::functional::unfold3d(x.detach(), kernel_size, dilation, padding, stride,
                                                         transpose_out), x.track_grad());
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                utils::my_n_tuple<3> output_size(parents[0]->detach().shape()[-3],
                                                 parents[0]->detach().shape()[-2],
                                                 parents[0]->detach().shape()[-1]);
                ::nt::functional::unfold3d_backward(grad, parents[0]->grad(), output_size,
                                                    kernel_size, dilation, padding, stride,
                                                    transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::unfoldnd(
        const TensorGrad &x, int64_t dim, utils::optional_list kernel_size,
        utils::optional_list dilation, utils::optional_list padding,
        utils::optional_list stride, bool transpose_out, bool test_mode) {
    TensorGrad result(::nt::functional::unfoldnd(x.detach(), dim, kernel_size, dilation, padding, stride,
                                                         transpose_out, test_mode), x.track_grad());
    result.track_tensors(x);
    result.create_backward_function(
            [dim, kernel_size, dilation, padding, stride, transpose_out, test_mode](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                std::vector<int64_t> output_size(dim);
                auto sh = parents[0]->detach().shape();
                for(int64_t i = 0; i < dim; ++i){
                    output_size[i] = parents[0]->detach().shape()[(-1 * dim) + i];
                }
                ::nt::functional::unfoldnd_backward(grad, parents[0]->grad(), dim, output_size,
                                                    kernel_size, dilation, padding, stride,
                                                    transpose_out, test_mode);
            });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::fold1d(const TensorGrad &x,
                                             Tensor::size_value_t output_size,
                                             Tensor::size_value_t kernel_size,
                                             Tensor::size_value_t dilation,
                                             Tensor::size_value_t padding,
                                             Tensor::size_value_t stride) {
    TensorGrad result(
            ::nt::functional::fold1d(x.detach(), output_size, kernel_size, dilation, padding, stride), x.track_grad());
    result.track_tensors(x);
    // it is coppied because the backward pass will go out of scope of this
    // function and so I dont want that memory to try to be referenced
    result.create_backward_function(
            [output_size, kernel_size, dilation, padding, stride](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                ::nt::functional::fold1d_backward(grad, parents[0]->grad(), output_size, kernel_size,
                                            dilation, padding, stride);
            });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::fold2d(const TensorGrad &x,
                                             utils::my_tuple output_size,
                                             utils::my_tuple kernel_size,
                                             utils::my_tuple dilation,
                                             utils::my_tuple padding,
                                             utils::my_tuple stride) {
    TensorGrad result(
            ::nt::functional::fold2d(x.detach(), output_size, kernel_size, dilation, padding, stride), x.track_grad());
    result.track_tensors(x);
    // it is coppied because the backward pass will go out of scope of this
    // function and so I dont want that memory to try to be referenced
    result.create_backward_function(
            [output_size, kernel_size, dilation, padding, stride](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                ::nt::functional::fold2d_backward(grad, parents[0]->grad(), output_size, kernel_size,
                                            dilation, padding, stride);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fold3d(const TensorGrad &x,
                                             utils::my_n_tuple<3> output_size,
                                             utils::my_n_tuple<3> kernel_size,
                                             utils::my_n_tuple<3> dilation,
                                             utils::my_n_tuple<3> padding,
                                             utils::my_n_tuple<3> stride) {
    TensorGrad result(
            ::nt::functional::fold3d(x.detach(), output_size, kernel_size, dilation, padding, stride), x.track_grad());
    result.track_tensors(x);
    // it is coppied because the backward pass will go out of scope of this
    // function and so I dont want that memory to try to be referenced
    result.create_backward_function(
            [output_size, kernel_size, dilation, padding, stride](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                ::nt::functional::fold3d_backward(grad, parents[0]->grad(), output_size, kernel_size,
                                            dilation, padding, stride);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::foldnd(const TensorGrad &x, int64_t dim,
                                             utils::optional_list output_size,
                                             utils::optional_list kernel_size,
                                             utils::optional_list dilation,
                                             utils::optional_list padding,
                                             utils::optional_list stride, bool test_mode) {
    TensorGrad result(
            ::nt::functional::foldnd(x.detach(), dim, output_size, kernel_size, dilation, padding, stride, test_mode), x.track_grad());
    result.track_tensors(x);
    // it is coppied because the backward pass will go out of scope of this
    // function and so I dont want that memory to try to be referenced
    result.create_backward_function(
            [dim, output_size, kernel_size, dilation, padding, stride, test_mode](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                if(parents[0]->detach().is_null() || !parents[0]->track_grad()){
                    parents[0]->accumulate_gradient(0);
                    return;
                }
                ::nt::functional::foldnd_backward(grad, parents[0]->grad(), dim, output_size, kernel_size,
                                            dilation, padding, stride, test_mode);
            });
    return std::move(result);
}

}
}
