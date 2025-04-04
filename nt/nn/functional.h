// this is a friend class to TensorGrad that holds the functional functions
// dedicated to tensorgrad this is meant to re-create the functional.h that is
// dedicated to the Tensor class

#ifndef _NT_TENSORGAD_FUNCTIONAL_H_
#define _NT_TENSORGAD_FUNCTIONAL_H_



#include "functional_class.h"

namespace nt {
namespace functional {

inline TensorGrad matmult(const TensorGrad &a, const TensorGrad &b, bool transpose_a = false, bool transpose_b = false) {
    return TensorGrad_Functional_Class::matmult(a, b, transpose_a, transpose_b);
}

inline TensorGrad matmult(const Tensor &a, const TensorGrad &b) {
    return TensorGrad_Functional_Class::matmult(a, b);
}

inline TensorGrad matmult(const TensorGrad &a, const Tensor &b) {
    return TensorGrad_Functional_Class::matmult(a, b);
}

inline TensorGrad
unfold1d(const TensorGrad &a, Tensor::size_value_t kernel_size,
         Tensor::size_value_t dilation = 1, Tensor::size_value_t padding = 0,
         Tensor::size_value_t stride = 1, bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold1d(
        a, kernel_size, dilation, padding, stride, transpose_out);
}

inline TensorGrad unfold(const TensorGrad &a, utils::my_tuple kernel_size,
                         utils::my_tuple dilation = 1,
                         utils::my_tuple padding = 0,
                         utils::my_tuple stride = 1,
                         bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold(a, kernel_size, dilation,
                                               padding, stride, transpose_out);
}

inline TensorGrad
unfold3d(const TensorGrad &a, utils::my_n_tuple<3> kernel_size,
         utils::my_n_tuple<3> dilation = 1, utils::my_n_tuple<3> padding = 0,
         utils::my_n_tuple<3> stride = 1, bool transpose_out = true) {
    return TensorGrad_Functional_Class::unfold3d(
        a, kernel_size, dilation, padding, stride, transpose_out);
}

inline TensorGrad fold(const TensorGrad &a, utils::my_tuple output_size,
                       utils::my_tuple kernel_size,
                       utils::my_tuple dilation = 1,
                       utils::my_tuple padding = 0,
                       utils::my_tuple stride = 1) {
    return TensorGrad_Functional_Class::fold(a, output_size, kernel_size,
                                             dilation, padding, stride);
}

inline TensorGrad conv1d(const TensorGrad &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv1d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv1d(const Tensor &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv1d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv1d(const TensorGrad &image, const Tensor &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv1d(image, kernel, stride,
                                                   padding, dilation, groups);
}


inline TensorGrad conv2d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv2d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv2d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv2d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv2d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv2d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv3d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv3d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv3d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv3d(image, kernel, stride,
                                                   padding, dilation, groups);
}

inline TensorGrad conv3d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> dilation = 1, int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv3d(image, kernel, stride,
                                                   padding, dilation, groups);
}


inline TensorGrad conv_transpose1d(const TensorGrad &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t output_padding = 0, int64_t dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose1d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose1d(const Tensor &image, const TensorGrad &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t output_padding = 0, int64_t dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose1d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose1d(const TensorGrad &image, const Tensor &kernel,
                         int64_t stride = 1, int64_t padding = 0,
                         int64_t output_padding = 0, int64_t dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose1d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}


inline TensorGrad conv_transpose2d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple output_padding = 0,utils::my_tuple dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose2d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose2d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple output_padding = 0,utils::my_tuple dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose2d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose2d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_tuple stride = 1, utils::my_tuple padding = 0,
                         utils::my_tuple output_padding = 0,utils::my_tuple dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose2d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose3d(const TensorGrad &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> output_padding = 0, utils::my_n_tuple<3> dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose3d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose3d(const Tensor &image, const TensorGrad &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0,
                         utils::my_n_tuple<3> output_padding = 0, utils::my_n_tuple<3> dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose3d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}

inline TensorGrad conv_transpose3d(const TensorGrad &image, const Tensor &kernel,
                         utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0, 
                         utils::my_n_tuple<3> output_padding = 0, utils::my_n_tuple<3> dilation = 1, 
                         int64_t groups = 1) {
    return TensorGrad_Functional_Class::conv_transpose3d(image, kernel, stride,
                                                padding, output_padding, dilation, groups);
}



inline TensorGrad sigmoid(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sigmoid(a);
}

inline TensorGrad clamp(const TensorGrad &a,
                        std::optional<int64_t> min = std::nullopt,
                        std::optional<int64_t> max = std::nullopt) {
    return TensorGrad_Functional_Class::clamp(a, min, max);
}

inline TensorGrad relu(const TensorGrad &a) {
    return TensorGrad_Functional_Class::relu(a);
}

inline TensorGrad var(const TensorGrad &a, utils::optional_list dim = nullptr,
                      int64_t correction = 1, bool keepdim = false) {
    return TensorGrad_Functional_Class::var(a, dim, correction, keepdim);
}

inline TensorGrad sqrt(const TensorGrad &a) {
    return TensorGrad_Functional_Class::sqrt(a);
}

inline TensorGrad invsqrt(const TensorGrad &a) {
    return TensorGrad_Functional_Class::invsqrt(a);
}

inline TensorGrad silu(const TensorGrad &a) {
    return TensorGrad_Functional_Class::silu(a);
}
inline TensorGrad gelu(const TensorGrad &a) {
    return TensorGrad_Functional_Class::gelu(a);
}
inline TensorGrad tanh(const TensorGrad &a) {
    return TensorGrad_Functional_Class::tanh(a);
}
inline TensorGrad tan(const TensorGrad &a) {
    return TensorGrad_Functional_Class::tan(a);
}
inline TensorGrad cat(std::vector<TensorGrad> tgs, int64_t dim = 0) {
    return TensorGrad_Functional_Class::cat(std::move(tgs), dim);
}

inline TensorGrad cat(TensorGrad tg, int64_t dim = 0) {
    return TensorGrad_Functional_Class::cat(std::move(tg), dim);
}

inline TensorGrad chunk(TensorGrad input, typename Tensor::size_value_t chunks,
                        int64_t dim = 0) {
    return TensorGrad_Functional_Class::chunk(std::move(input), chunks, dim);
}


inline TensorGrad split(TensorGrad input, typename Tensor::size_value_t split_size, int64_t dim){
    return TensorGrad_Functional_Class::split(std::move(input), split_size, dim);
}


inline TensorGrad split(TensorGrad input, std::vector<typename Tensor::size_value_t> split_sections, int64_t dim){
    return TensorGrad_Functional_Class::split(std::move(input), std::move(split_sections), dim);
    
}

inline TensorGrad stack(TensorGrad input, int64_t dim){
    return TensorGrad_Functional_Class::stack(std::move(input), dim);
}

inline TensorGrad stack(std::vector<TensorGrad> input, int64_t dim){
    return TensorGrad_Functional_Class::stack(std::move(input), dim);
}


inline TensorGrad log(const TensorGrad &a) {
    return TensorGrad_Functional_Class::log(a);
}


inline TensorGrad dropout(const TensorGrad &input, double p) {
    return TensorGrad_Functional_Class::dropout(input, p);
}

inline TensorGrad abs(const TensorGrad &input){
    return TensorGrad_Functional_Class::abs(input);
}

inline TensorGrad softplus(const TensorGrad &input, Scalar beta=1.0, Scalar threshold=20.0){
    return TensorGrad_Functional_Class::softplus(input, beta, threshold);
}

template<typename T, typename... Args,
         std::enable_if_t<std::is_same_v<std::decay_t<T>, TensorGrad>, int>>
inline TensorGrad list(T&& first, Args&&... rest){
	static_assert(utils::is_all_same_v<std::decay_t<T>, std::decay_t<Args>...>, 
                  "Expected to make a list of all TensorGrads");

    // create the result TensorGrad
    TensorGrad result(list(first.tensor, (rest.tensor) ...),
                      first.grad_required);

    bool track_grad = first.do_track_grad;
    bool require_grad = first.grad_required;

    // ensure consistency for tracking and requiring gradients
    ((utils::throw_exception(std::forward<Args>(rest).do_track_grad ==
                                 track_grad,
                             "Expected consistent track_grad values")),
     ...);
    ((utils::throw_exception(std::forward<Args>(rest).grad_required ==
                                 require_grad,
                             "Expected consistent grad_required values")),
     ...);

    // update track_grad and grad_required flags
    if (!require_grad) {
        track_grad = false;
    }
    if (!track_grad) {
        result.do_track_grad = false;
        return result; // return directly if tracking is not needed
    }



    // initialize grads if not already set
    if (first.grad == nullptr) {
        first.grad =
            make_intrusive<tensor_holder>(functional::zeros_like(first.tensor));
    }
    ((rest.grad = (rest.grad == nullptr)
                      ? make_intrusive<tensor_holder>(
                            functional::zeros_like(rest.tensor))
                      : rest.grad),
     ...);
    //create the gradient for the result
    result.grad = make_intrusive<tensor_holder>(
        list(first.grad->tensor, (rest.grad->tensor) ...));
    
    // set up parent references
    // this also automatically tracks children
    result.track_tensors(first, rest...);

    return result;
}

} // namespace functional
} // namespace nt

#endif // _NT_TENSORGAD_FUNCTIONAL_H_
