/*

This is a header file that contains everything in a more user-friendly way
Everything defined in the nt:: header that is a function is applicable to bot the nt::Tensor and the nt::TensorGrad class
    - Unless specified to return an nt::Tensor
All functions (except for a select few with reasoning below) also have named parameters
This is the header file that is meant to be included to contain all of the Tensor functionality and the AI functionality
Everything inside of the functional namespace is meant to stay as-is

*/

#ifndef NT_REFLECTED_HEADER_FILE_H__
#define NT_REFLECTED_HEADER_FILE_H__

#include "functional/functional.h"
#include "nn/functional.h"
#include "reflection/named_parameters/named_parameters.hpp"
#include "reflection/named_parameters/src/initializer_list.hpp"
#include "utils/macros.h"
#include "utils/optional_any_tuple.h"
#include "utils/optional_tensor.h"
#include "utils/optional_tensorgrad.h"
#include "utils/always_inline_macro.h"

namespace nt{


#define ADD_UNDERSCORE(name) name##_

#define NT_MAKE_NAMED_ACTIVATION_FUNCTION(name)\
NT_MAKE_NAMED_PARAMETER_FUNCTION_(name)\
    ntarg_(input)\
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::name) \
\
NT_MAKE_NAMED_PARAMETER_FUNCTION_(ADD_UNDERSCORE(name)) \
    ntarg_(input) \
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::ADD_UNDERSCORE(name)) \

    

NT_MAKE_NAMED_ACTIVATION_FUNCTION(sigmoid)
NT_MAKE_NAMED_ACTIVATION_FUNCTION(sqrt)
NT_MAKE_NAMED_ACTIVATION_FUNCTION(invsqrt)
NT_MAKE_NAMED_ACTIVATION_FUNCTION(abs)
NT_MAKE_NAMED_ACTIVATION_FUNCTION(relu)
NT_MAKE_NAMED_ACTIVATION_FUNCTION(gelu)
NT_MAKE_NAMED_ACTIVATION_FUNCTION(silu)





//activation_functions.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_(pow)
    ntarg_(input),
    ntarg_(exponent)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::pow)


NT_MAKE_NAMED_PARAMETER_FUNCTION_(softplus)
    ntarg_(input),
    ntarg_(beta) = 1.0,
    ntarg_(threshold) = 20.0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::softplus);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(pow_)
    ntarg_(input),
    ntarg_(exponent)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::pow_)


NT_MAKE_NAMED_PARAMETER_FUNCTION_(softplus_)
    ntarg_(input),
    ntarg_(beta) = 1.0,
    ntarg_(threshold) = 20.0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::softplus_);

//colim_transforms
// #define NT_NAMED_MY_TUPLE NT_NAMED_CONVERTIBLE(utils::my_tuple)
// #define NT_NAMED_MY_N_TUPLE(num) NT_NAMED_CONVERTIBLE(utils::my_n_tuple<num>)



//unfortunately, c++ does not allow initializer lists as a parameter pack expansion
//therefore, this has to be done [for now, maybe this can be manually made into a function that could expect it]

namespace functional_details{

template<typename... Args>
NT_ALWAYS_INLINE auto unfoldnd(Args&&... args){
    static_assert(sizeof...(args) == 7,
                  "INTERNAL ERROR, EXPECTED UNFOLDND TO GET 6 ELEMENTS");
    ::nt::functional::unfoldnd(std::forward<Args>(args)..., false); // makes test_mode = false
}

template<typename... Args>
NT_ALWAYS_INLINE auto foldnd(Args&&... args){
    static_assert(sizeof...(args) == 7,
                  "INTERNAL ERROR, EXPECTED FOLDND TO GET 6 ELEMENTS");
    ::nt::functional::foldnd(std::forward<Args>(args)..., false); // makes test_mode = false
}


}


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fold1d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fold1d, 6, functional::fold1d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fold2d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fold2d, 6, functional::fold2d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fold3d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fold3d, 6, functional::fold3d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(foldnd)
    ntarg_(input),
    ntarg_(dim),
    ntarg_(output_size),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(foldnd, 7, functional_details::foldnd);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(unfold1d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1,
    ntarg_(transpose_out) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::unfold1d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unfold2d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1,
    ntarg_(transpose_out) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unfold2d, 6, functional::unfold2d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unfold3d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1,
    ntarg_(transpose_out) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unfold3d, 6, functional::unfold3d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unfoldnd)
    ntarg_(input),
    ntarg_(dim),
    ntarg_(kernel_size),
    ntarg_(dilation) = 1,
    ntarg_(padding) = 0,
    ntarg_(stride) = 1,
    ntarg_(transpose_out) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unfoldnd, 7, functional_details::unfoldnd);


//combinations

NT_MAKE_NAMED_PARAMETER_FUNCTION_(combinations)
    ntarg_(vec),
    ntarg_(r),
    ntarg_(start) = 0
NT_FINISH_NAMED_PARAMETER_FUNCTION_(functional::combinations)


//combine

namespace functional_details{

NT_ALWAYS_INLINE Tensor cat(const std::initializer_list<Tensor>& il, int64_t dim){
    return ::nt::functional::cat(Tensor(il), dim);
}

NT_ALWAYS_INLINE Tensor cat(const Tensor& t, int64_t dim){
    return ::nt::functional::cat(t, dim);
}

NT_ALWAYS_INLINE Tensor cat(std::vector<Tensor> ts, int64_t dim){
    return ::nt::functional::cat(std::move(ts), dim);
}

NT_ALWAYS_INLINE TensorGrad cat(const TensorGrad& tg, int64_t dim){ return ::nt::functional::cat(tg); }
NT_ALWAYS_INLINE TensorGrad cat(std::vector<TensorGrad> tgs, int64_t dim){return ::nt::functional::cat(std::move(tgs), dim);}


NT_ALWAYS_INLINE Tensor stack(const std::initializer_list<Tensor>& il, int64_t dim){
    return ::nt::functional::stack(Tensor(il), dim);
}

NT_ALWAYS_INLINE Tensor stack(const Tensor& t, int64_t dim){
    return ::nt::functional::stack(t, dim);
}

NT_ALWAYS_INLINE Tensor stack(std::vector<Tensor> ts, int64_t dim){
    return ::nt::functional::stack(std::move(ts), dim);
}

NT_ALWAYS_INLINE TensorGrad stack(const TensorGrad& tgs, int64_t dim){return ::nt::functional::stack(tgs, dim);}
NT_ALWAYS_INLINE TensorGrad stack(std::vector<TensorGrad> tgs, int64_t dim){return ::nt::functional::stack(std::move(tgs), dim);}

}

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(cat)
    ntarg_(tensors),
    ntarg_(dim) = 0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(cat, 2, functional_details::cat);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(stack)
    ntarg_(tensors),
    ntarg_(dim) = 0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(stack, 2, functional_details::stack);

template<typename T, typename... Args,
         typename std::enable_if<std::is_same<std::decay_t<T>, Tensor>::value, int>::type = 0>
NT_ALWAYS_INLINE Tensor list(T&& first, Args&&... rest){return functional::list(std::forward<T>(first), std::forward<Args>(rest)...);}

template<typename T, typename... Args,
         typename std::enable_if<std::is_same<std::decay_t<T>, TensorGrad>::value, int>::type = 0>
NT_ALWAYS_INLINE Tensor list(T&& first, Args&&... rest){return functional::list(std::forward<T>(first), std::forward<Args>(rest)...);}



//compare.h

#define NT_DUAL_COMPARE_OP(name)\
NT_ALWAYS_INLINE Tensor name(const Tensor& a, const Tensor& b){return functional::name(a, b);}\
NT_ALWAYS_INLINE Tensor name(const TensorGrad& a, const Tensor& b){return functional::name(a.detach(), b);}\
NT_ALWAYS_INLINE Tensor name(const Tensor& a, const TensorGrad& b){return functional::name(a, b.detach());}\
NT_ALWAYS_INLINE Tensor name(const TensorGrad& a, const TensorGrad& b){return functional::name(a.detach(), b.detach());}\


NT_DUAL_COMPARE_OP(equal)
NT_DUAL_COMPARE_OP(not_equal)
NT_DUAL_COMPARE_OP(less_than)
NT_DUAL_COMPARE_OP(greater_than)
NT_DUAL_COMPARE_OP(less_than_equal)
NT_DUAL_COMPARE_OP(greater_than_equal)


#undef NT_DUAL_COMPARE_OP

NT_ALWAYS_INLINE bool any(const Tensor& a){return functional::any(a);}
NT_ALWAYS_INLINE bool all(const Tensor& a){return functional::all(a);}
NT_ALWAYS_INLINE bool none(const Tensor& a){return functional::none(a);}
NT_ALWAYS_INLINE Tensor any(const Tensor& a, int64_t dim){return functional::any(a, dim);}
NT_ALWAYS_INLINE Tensor all(const Tensor& a, int64_t dim){return functional::all(a, dim);}
NT_ALWAYS_INLINE Tensor where(const Tensor& t){return functional::where(t);}
NT_ALWAYS_INLINE Tensor where(const TensorGrad& t){return functional::where(t.detach());}
NT_ALWAYS_INLINE int64_t count(const Tensor& t){return functional::count(t);}
NT_ALWAYS_INLINE int64_t count(const TensorGrad& tg){return functional::count(tg.detach());}

namespace functional_details{
NT_ALWAYS_INLINE Tensor isnan(const Tensor& a){return functional::isnan(a);}
NT_ALWAYS_INLINE Tensor isnan(const TensorGrad& a){return functional::isnan(a.detach());}
NT_ALWAYS_INLINE bool allclose(const Tensor& input, const Tensor& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::allclose(input, other, rtol, atol, equal_nan);
}
NT_ALWAYS_INLINE bool allclose(const TensorGrad& input, const Tensor& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::allclose(input.detach(), other, rtol, atol, equal_nan);
}
NT_ALWAYS_INLINE bool allclose(const Tensor& input, const TensorGrad& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::allclose(input, other.detach(), rtol, atol, equal_nan);
}
NT_ALWAYS_INLINE bool allclose(const TensorGrad& input, const TensorGrad& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::allclose(input.detach(), other.detach(), rtol, atol, equal_nan);
}
NT_ALWAYS_INLINE Tensor isclose(const Tensor& input, const Tensor& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::isclose(input, other, rtol, atol, equal_nan);
}

NT_ALWAYS_INLINE Tensor isclose(const TensorGrad& input, const Tensor& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::isclose(input.detach(), other, rtol, atol, equal_nan);
}
NT_ALWAYS_INLINE Tensor isclose(const Tensor& input, const TensorGrad& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::isclose(input, other.detach(), rtol, atol, equal_nan);
}
NT_ALWAYS_INLINE Tensor isclose(const TensorGrad& input, const TensorGrad& other, Scalar rtol, Scalar atol, bool equal_nan){
    return ::nt::functional::isclose(input.detach(), other.detach(), rtol, atol, equal_nan);
}

}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(isnan)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::isnan);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(allclose)
    ntarg_(input),
    ntarg_(other),
    ntarg_(rtol) = float(1e-5),
    ntarg_(atol) = float(1e-8),
    ntarg_(equal_nan) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::allclose);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(isclose)
    ntarg_(input),
    ntarg_(other),
    ntarg_(rtol) = float(1e-5),
    ntarg_(atol) = float(1e-8),
    ntarg_(equal_nan) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::isclose);

namespace functional_details{
NT_ALWAYS_INLINE int64_t amount_of(const Tensor& t, Scalar val){return ::nt::functional::amount_of(t, val);}
NT_ALWAYS_INLINE int64_t amount_of(const TensorGrad& t, Scalar val){return ::nt::functional::amount_of(t.detach(), val);}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(amount_of)
    ntarg_(input),
    ntarg_(val) = 0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::amount_of);


//complex.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_(real)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::real);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(imag)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::imag);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(to_complex_from_real)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::to_complex_from_real);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(to_complex_from_imag)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::to_complex_from_imag);


//conv.h

namespace functional_details{

NT_ALWAYS_INLINE Tensor conv1d(const Tensor& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv1d(image, kernel, stride, padding, dilation, groups, nullptr, nullptr);
}

NT_ALWAYS_INLINE Tensor conv2d(const Tensor& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv2d(image, kernel, stride, padding, dilation, groups, nullptr, nullptr);
}

NT_ALWAYS_INLINE Tensor conv3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation,int64_t groups){
    return ::nt::functional::conv3d(image, kernel, stride, padding, dilation, groups, nullptr, nullptr);
}

NT_ALWAYS_INLINE Tensor convnd(const Tensor& image, const Tensor& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list dilation,int64_t groups){
    return ::nt::functional::convnd(image, kernel, dim, stride, padding, dilation, groups, nullptr, nullptr);
}

NT_ALWAYS_INLINE Tensor conv_transpose1d(const Tensor& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv_transpose1d(image, kernel, stride, padding, output_padding, dilation, groups, nullptr, nullptr);
}

NT_ALWAYS_INLINE Tensor conv_transpose2d(const Tensor& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv_transpose2d(image, kernel, stride, padding, output_padding, dilation, groups, nullptr, nullptr);
}

NT_ALWAYS_INLINE Tensor conv_transpose3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    return ::nt::functional::conv_transpose3d(image, kernel, stride, padding, output_padding, dilation, groups, nullptr, nullptr);
}

NT_ALWAYS_INLINE Tensor conv_transposend(const Tensor& image, const Tensor& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list output_padding, utils::optional_list dilation, int64_t groups){
    return ::nt::functional::conv_transposend(image, kernel, dim, stride, padding, output_padding, dilation, groups, nullptr, nullptr);
}


NT_ALWAYS_INLINE TensorGrad conv1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv1d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv2d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    return ::nt::functional::conv3d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad convnd(const TensorGrad& image, const TensorGrad& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list dilation, int64_t groups){
    return ::nt::functional::convnd(image, kernel, dim, stride, padding, dilation, groups);
}
NT_ALWAYS_INLINE TensorGrad conv_transpose1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv_transpose1d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transpose2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv_transpose2d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transpose3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    return ::nt::functional::conv_transpose3d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transposend(const TensorGrad& image, const TensorGrad& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list output_padding, utils::optional_list dilation, int64_t groups){
    return ::nt::functional::conv_transposend(image, kernel, dim, stride, padding, output_padding, dilation, groups);
}


NT_ALWAYS_INLINE TensorGrad conv1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv1d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv2d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    return ::nt::functional::conv3d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad convnd(const Tensor& image, const TensorGrad& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list dilation, int64_t groups){
    return ::nt::functional::convnd(image, kernel, dim, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transpose1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv_transpose1d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transpose2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv_transpose2d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transpose3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    return ::nt::functional::conv_transpose3d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transposend(const Tensor& image, const TensorGrad& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list output_padding, utils::optional_list dilation, int64_t groups){
    return ::nt::functional::conv_transposend(image, kernel, dim, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv1d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv2d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    return ::nt::functional::conv3d(image, kernel, stride, padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad convnd(const TensorGrad& image, const Tensor& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list dilation, int64_t groups){
    return ::nt::functional::convnd(image, kernel, dim, stride, padding, dilation, groups);
}


NT_ALWAYS_INLINE TensorGrad conv_transpose1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
    return ::nt::functional::conv_transpose1d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transpose2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
    return ::nt::functional::conv_transpose2d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transpose3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    return ::nt::functional::conv_transpose3d(image, kernel, stride, padding, output_padding, dilation, groups);
}

NT_ALWAYS_INLINE TensorGrad conv_transposend(const TensorGrad& image, const Tensor& kernel, int64_t dim, utils::optional_list stride, utils::optional_list padding, utils::optional_list output_padding, utils::optional_list dilation, int64_t groups){
    return ::nt::functional::conv_transposend(image, kernel, dim, stride, padding, output_padding, dilation, groups);
}

}


NT_MAKE_NAMED_PARAMETER_FUNCTION_(conv1d)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::conv1d);




NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv2d)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv2d, 6, functional_details::conv2d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv3d)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv3d, 6, functional_details::conv3d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(convnd)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(dim),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(convnd, 7, functional_details::convnd);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(conv_transpose1d)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(output_padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::conv_transpose1d);




NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv_transpose2d)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(output_padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv_transpose2d, 7, functional_details::conv_transpose2d);


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv_transpose3d)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(output_padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv_transpose3d, 7, functional_details::conv_transpose3d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv_transposend)
    ntarg_(image),
    ntarg_(kernel),
    ntarg_(dim),
    ntarg_(stride) = 1,
    ntarg_(padding) = 0,
    ntarg_(output_padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(groups) = 1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(conv_transposend, 8, functional_details::conv_transposend);

//Convert.h 
//floating_, complex_, integer_, and unsigned_ not included here


NT_MAKE_NAMED_PARAMETER_FUNCTION_(to)
    ntarg_(input),
    ntarg_(type) //named type because next iteration will add DeviceType
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::to);



//dilate.h

namespace functional_details{
    NT_ALWAYS_INLINE Tensor undilate(const Tensor& x, utils::optional_any_tuple<Tensor::size_value_t> tup){
        return ::nt::functional::undilate(x, tup.get_vals());
    }
    NT_ALWAYS_INLINE Tensor undilate_(const Tensor& x, utils::optional_any_tuple<Tensor::size_value_t> tup, bool test){
        return ::nt::functional::undilate_(x, tup.get_vals(), test);
    }
    NT_ALWAYS_INLINE Tensor dilate(const Tensor& x, utils::optional_any_tuple<Tensor::size_value_t> tup, bool test){
        return ::nt::functional::dilate(x, tup.get_vals(), test);
    }
    NT_ALWAYS_INLINE TensorGrad undilate(const TensorGrad& x, utils::optional_any_tuple<Tensor::size_value_t> tup){
        return ::nt::functional::dilate(x, tup.get_vals());
    }
    NT_ALWAYS_INLINE TensorGrad undilate_(const TensorGrad& x, utils::optional_any_tuple<Tensor::size_value_t> tup, bool test){
        return ::nt::functional::undilate_(x, tup.get_vals(), test);
    }
    NT_ALWAYS_INLINE TensorGrad dilate(const TensorGrad& x, utils::optional_any_tuple<Tensor::size_value_t> tup, bool test){
        return ::nt::functional::dilate(x, tup.get_vals(), test);
    }
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(dilate)
    ntarg_(input),
    ntarg_(dilation),
    ntarg_(test) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(dilate, 3, functional_details::dilate);


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(undilate)
    ntarg_(input),
    ntarg_(dilation)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(undilate, 2, functional_details::undilate);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(undilate_)
    ntarg_(input),
    ntarg_(dilation),
    ntarg_(test) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(undilate_, 3, functional_details::undilate_);


//dropout.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_(dropout)
    ntarg_(input),
    ntarg_(ratio)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::dropout);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(dropout2d)
    ntarg_(input),
    ntarg_(ratio)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::dropout2d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(dropout3d)
    ntarg_(input),
    ntarg_(ratio)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::dropout3d);


//fill.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(zeros)
    ntarg_(size),
    ntarg_(dtype) = DType::Float32
NT_FINISH_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(zeros, 2,functional::zeros)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(ones)
    ntarg_(size),
    ntarg_(dtype) = DType::Float32
NT_FINISH_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(ones, 2, functional::ones)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(nums)
    ntarg_(size),
    ntarg_(num),
    ntarg_(dtype) = DType::Float32
NT_FINISH_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(nums, 3, functional::nums)


NT_MAKE_NAMED_PARAMETER_FUNCTION_(zeros_like)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::zeros_like);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(ones_like)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::ones_like);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(nums_like)
    ntarg_(input),
    ntarg_(num)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::nums_like);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(arange)
    ntarg_(size),
    ntarg_(dtype) = DType::Float32,
    ntarg_(start) = 0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(arange, 3, functional::arange)


NT_MAKE_NAMED_PARAMETER_FUNCTION_(fill_diagonal_)
    ntarg_(input),
    ntarg_(value) 
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::fill_diagonal_)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(fill_)
    ntarg_(input),
    ntarg_(value) 
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::fill_)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(set_)
    ntarg_(input),
    ntarg_(tensor) 
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::set_)

//flip.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(flip)
    ntarg_(input),
    ntarg_(list) = nullptr
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(flip, 2, functional::flip);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(flip_view)
    ntarg_(input),
    ntarg_(list) = nullptr
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(flip_view, 2, functional::flip_view)


//fused.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_(fused_multiply_add)
    ntarg_(input),
    ntarg_(tensor1),
    ntarg_(tensor2)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::fused_multiply_add)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(fused_multiply_add_)
    ntarg_(input),
    ntarg_(tensor1),
    ntarg_(tensor2)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::fused_multiply_add_)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(fused_multiply_subtract)
    ntarg_(input),
    ntarg_(tensor1),
    ntarg_(tensor2)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::fused_multiply_subtract)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(fused_multiply_subtract_)
    ntarg_(input),
    ntarg_(tensor1),
    ntarg_(tensor2)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::fused_multiply_subtract_)


//index.h
NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(at)
    ntarg_(input),
    ntarg_(idx)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(at, 2, functional::at);


namespace functional_details{
NT_ALWAYS_INLINE TensorGrad at_tensor_split(const TensorGrad& input, const TensorGrad& idx, Tensor::size_value_t splitting, utils::optional_tensorgrad tg){
    if(tg.has_value()){
        return functional::at_tensor_split(input, idx, splitting, tg.value()); 
    }else{
        return functional::at_tensor_split(input, idx, splitting); 
    }
}
NT_ALWAYS_INLINE TensorGrad at_tensor_split(const TensorGrad & input, const Tensor & idx, Tensor::size_value_t splitting, utils::optional_tensorgrad tg){
    if(tg.has_value()){
        return functional::at_tensor_split(input, idx, splitting, tg.value()); 
    }else{
        return functional::at_tensor_split(input, idx, splitting); 
    }
 
}
NT_ALWAYS_INLINE Tensor at_tensor_split(const Tensor & input, const Tensor & idx, Tensor::size_value_t splitting, utils::optional_tensor t){
    if(t.has_value()){
        return functional::at_tensor_split(input, idx, splitting, t.value()); 
    }else{
        return functional::at_tensor_split(input, idx, splitting); 
    }
 
}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(at_tensor_split)
    ntarg_(input),
    ntarg_(idx),
    ntarg_(splitting),
    ntarg_(output) = nullptr
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::at_tensor_split);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(index_select)
    ntarg_(input),
    ntarg_(dim),
    ntarg_(idx)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::index_select);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(index_except)
    ntarg_(input),
    ntarg_(dim),
    ntarg_(idx)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::index_except);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(select)
    ntarg_(input),
    ntarg_(dim),
    ntarg_(idx)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::select);



//matmult.h

namespace functional_details{
NT_ALWAYS_INLINE TensorGrad matmult(const TensorGrad & a, const Tensor & b, utils::optional_tensorgrad tg, bool transpose_a, bool transpose_b){
    if(tg.has_value()){
        return ::nt::functional::matmult(a, b, tg.value(), transpose_a, transpose_b); 
    }else{
        return ::nt::functional::matmult(a, b, transpose_a, transpose_b); 
    }
 
}
NT_ALWAYS_INLINE Tensor matmult(const Tensor & a, const Tensor & b, utils::optional_tensor t, bool transpose_a, bool transpose_b){
    if(t.has_value()){
        return ::nt::functional::matmult(a, b, t.value(), transpose_a, transpose_b); 
    }else{
        return ::nt::functional::matmult(a, b, transpose_a, transpose_b); 
    }

 
}
 
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(matmult)
    ntarg_(input),
    ntarg_(other),
    ntarg_(out) = nullptr,
    ntarg_(transpose_a) = false,
    ntarg_(transpose_b) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::matmult);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(linear)
    ntarg_(input),
    ntarg_(weight),
    ntarg_(bias),
    ntarg_(transpose_a) = false,
    ntarg_(transpose_b) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::linear);


namespace functional_details{
NT_ALWAYS_INLINE Tensor one_hot(const TensorGrad& input, int64_t num_classes){
    return ::nt::functional::one_hot(input.detach(), num_classes);
}
NT_ALWAYS_INLINE Tensor one_hot(const Tensor& input, int64_t num_classes){
    return ::nt::functional::one_hot(input, num_classes);
}


NT_ALWAYS_INLINE Tensor meshgrid(const Tensor& input, const Tensor& other){
    return ::nt::functional::meshgrid(input, other);
}

NT_ALWAYS_INLINE Tensor meshgrid(const TensorGrad& input, const TensorGrad& other){
    return ::nt::functional::meshgrid(input.detach(), other.detach());
}

NT_ALWAYS_INLINE Tensor meshgrid(const TensorGrad& input, const Tensor& other){
    return ::nt::functional::meshgrid(input.detach(), other);
}

NT_ALWAYS_INLINE Tensor meshgrid(const Tensor& input, const TensorGrad& other){
    return ::nt::functional::meshgrid(input, other.detach());
}

}

//mesh.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_(one_hot)
    ntarg_(input),
    ntarg_(num_classes) = -1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::one_hot);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(meshgrid)
    ntarg_(input),
    ntarg_(other)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::meshgrid);


//min_max.h

namespace functional_details{
NT_ALWAYS_INLINE result_types::max<Tensor, Tensor> max(const Tensor& input, utils::optional_list dim = nullptr, bool keepdim = false){
    return functional::max(input, dim, keepdim);
}

NT_ALWAYS_INLINE result_types::max<TensorGrad, Tensor> max(const TensorGrad& input, utils::optional_list dim = nullptr, bool keepdim = false){
    return functional::max(input, dim, keepdim);
}

NT_ALWAYS_INLINE result_types::max<Tensor, Tensor> min(const Tensor& input, utils::optional_list dim = nullptr, bool keepdim = false){
    return functional::min(input, dim, keepdim);
}

NT_ALWAYS_INLINE result_types::max<TensorGrad, Tensor> min(const TensorGrad& input, utils::optional_list dim = nullptr, bool keepdim = false){
    return functional::min(input, dim, keepdim);
}

NT_ALWAYS_INLINE Tensor argmax(const Tensor& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false){
    if(dim.has_value()){
        return functional::argmax(input, dim.value(), keepdim);
    }
    return functional::argmax(input);
}


NT_ALWAYS_INLINE Tensor argmin(const Tensor& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false){
    if(dim.has_value()){
        return functional::argmin(input, dim.value(), keepdim);
    }
    return functional::argmin(input);
}

NT_ALWAYS_INLINE Tensor argmax(const TensorGrad& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false){
    if(dim.has_value()){
        return functional::argmax(input.detach(), dim.value(), keepdim);
    }
    return functional::argmax(input.detach());
}

NT_ALWAYS_INLINE Tensor argmin(const TensorGrad& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false){
    if(dim.has_value()){
        return functional::argmin(input.detach(), dim.value(), keepdim);
    }
    return functional::argmin(input.detach());
}


NT_ALWAYS_INLINE Tensor max_indices(const Tensor& input, utils::optional_list dim = nullptr, utils::optional_tensor indices = nullptr){
    if(indices.has_value()){
        return functional::max_indices(input, indices.value(), dim);
    }else{
        return functional::max_indices(input, dim);
    }
}

NT_ALWAYS_INLINE Tensor min_indices(const Tensor& input, utils::optional_list dim = nullptr, utils::optional_tensor indices = nullptr){
    if(indices.has_value()){
        return functional::min_indices(input, indices.value(), dim);
    }else{
        return functional::min_indices(input, dim);
    }
}


NT_ALWAYS_INLINE Tensor max_indices(const TensorGrad& input, utils::optional_list dim = nullptr, utils::optional_tensor indices = nullptr){
    if(indices.has_value()){
        return functional::max_indices(input.detach(), indices.value(), dim);
    }else{
        return functional::max_indices(input.detach(), dim);
    }
}

NT_ALWAYS_INLINE Tensor min_indices(const TensorGrad& input, utils::optional_list dim = nullptr, utils::optional_tensor indices = nullptr){
    if(indices.has_value()){
        return functional::min_indices(input.detach(), indices.value(), dim);
    }else{
        return functional::min_indices(input.detach(), dim);
    }
}


}


namespace functional {
namespace min_max_detail {

inline void to_vec_tensor_sub(std::vector<Tensor> &out_tensors) { ; }

template <typename T, typename... Args>
inline void to_vec_tensor_sub(std::vector<Tensor> &out_tensors, T &&arg,
                              Args &&...args) {
    static_assert(std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < ::nt::type_traits::remove_cvref_t < T >>,
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    if constexpr (std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor>) {
        out_tensors.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_tensor_sub(out_tensors, std::forward<Args &&>(args)...);
}

template <typename T, typename... Args>
inline std::vector<Tensor> to_vec_tensor(T &&arg, Args &&...args) {
    static_assert(std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < ::nt::type_traits::remove_cvref_t < T >>, 
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    std::vector<Tensor> out_tensors;
    out_tensors.reserve(sizeof...(Args) + 1);
    if constexpr (std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor>) {
        out_tensors.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_tensor_sub(out_tensors, std::forward<Args &&>(args)...);
    return std::move(out_tensors);
}

inline void to_vec_tensor_grad_sub(std::vector<TensorGrad> &out_tensors) { ; }

template <typename T, typename... Args>
inline void to_vec_tensor_grad_sub(std::vector<TensorGrad> &out_tensors, T &&arg,
                              Args &&...args) {
    static_assert(std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < ::nt::type_traits::remove_cvref_t < T >>,
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    if constexpr (std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad>) {
        out_tensors.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_tensor_grad_sub(out_tensors, std::forward<Args &&>(args)...);
}

template <typename T, typename... Args>
inline std::vector<TensorGrad> to_vec_tensor_grad(T &&arg, Args &&...args) {
    static_assert(std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < ::nt::type_traits::remove_cvref_t < T >>, 
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    std::vector<TensorGrad> out_tensors;
    out_tensors.reserve(sizeof...(Args) + 1);
    if constexpr (std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad>) {
        out_tensors.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_tensor_grad_sub(out_tensors, std::forward<Args &&>(args)...);
    return std::move(out_tensors);
}

inline void to_vec_scalar_sub(std::vector<Scalar> &out_scalarss) { ; }

template <typename T, typename... Args>
inline void to_vec_scalar_sub(std::vector<Scalar> &out_scalars, T &&arg,
                              Args &&...args) {
    static_assert(std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < ::nt::type_traits::remove_cvref_t < T >>, 
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    if constexpr (std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    } else if constexpr (utils::is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_scalar_sub(out_scalars, std::forward<Args &&>(args)...);
}

template <typename T, typename... Args>
inline std::vector<Scalar> to_vec_scalar(T &&arg, Args &&...args) {
    static_assert(std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, TensorGrad> ||
                  std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < ::nt::type_traits::remove_cvref_t < T >>, 
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    std::vector<Scalar> out_scalars;
    out_scalars.reserve(sizeof...(Args) + 1);
    if constexpr (std::is_same_v<::nt::type_traits::remove_cvref_t<T>, Scalar>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    } else if constexpr (utils::is_scalar_value_v<::nt::type_traits::remove_cvref_t<T>>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_scalar_sub(out_scalars, std::forward<Args &&>(args)...);
    return std::move(out_scalars);
}
} // namespace min_max_detail

// // can take arbitraty scalars and tensors and find the max
// template <typename... Args> inline Tensor maximum(Args &&...args) {
//     std::vector<Tensor> tensors =
//         min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
//     std::vector<Scalar> scalars =
//         min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);

//     if (scalars.size() == 0) {
//         return maximum(std::move(tensors));
//     } else if (tensors.size() == 0) {
//         Tensor out({1}, scalars[0].type());
//         out = maximum(std::move(scalars));
//         return std::move(out);
//     }
//     Scalar max_s = maximum(std::move(scalars));
//     return maximum(std::move(tensors), max_s);
// }

// // can take arbitraty scalars and tensors and find the min
// template <typename... Args> inline Tensor minimum(Args &&...args) {
//     std::vector<Tensor> tensors =
//         min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
//     std::vector<Scalar> scalars =
//         min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);

//     if (scalars.size() == 0) {
//         return minimum(std::move(tensors));
//     } else if (tensors.size() == 0) {
//         Tensor out({1}, scalars[0].type());
//         out = minimum(std::move(scalars));
//         return std::move(out);
//     }
//     Scalar min_s = minimum(std::move(scalars));
//     return minimum(std::move(tensors), min_s);
// }


} // namespace functional


template<typename... Args>
NT_ALWAYS_INLINE auto maximum(Args&&... args){
    if constexpr (utils::contains_decayed_type<TensorGrad, Args...>::value){
        std::vector<TensorGrad> tensor_grads =
            functional::min_max_detail::to_vec_tensor_grad(std::forward<Args &&>(args)...);
        std::vector<Tensor> tensors =
            functional::min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
        std::vector<Scalar> scalars =
            functional::min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);
        return functional::maximum(tensor_grads, tensors, scalars);
    }else if constexpr (utils::contains_decayed_type<Tensor, Args...>::value){
        std::vector<Tensor> tensors =
            functional::min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
        std::vector<Scalar> scalars =
            functional::min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);
        if (scalars.size() == 0) {
            return functional::maximum(std::move(tensors));
        } 
        Scalar max_s = functional::maximum(std::move(scalars));
        return functional::maximum(std::move(tensors), max_s);
    }else{
        std::vector<Scalar> scalars =
            functional::min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);
        return functional::maximum(std::move(scalars)); 
    }
}

template<typename... Args>
NT_ALWAYS_INLINE auto minimum(Args&&... args){
    if constexpr (utils::contains_decayed_type<TensorGrad, Args...>::value){
        std::vector<TensorGrad> tensor_grads =
            functional::min_max_detail::to_vec_tensor_grad(std::forward<Args &&>(args)...);
        std::vector<Tensor> tensors =
            functional::min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
        std::vector<Scalar> scalars =
            functional::min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);
        return functional::minimum(tensor_grads, tensors, scalars);
    }else if constexpr (utils::contains_decayed_type<Tensor, Args...>::value){
        std::vector<Tensor> tensors =
            functional::min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
        std::vector<Scalar> scalars =
            functional::min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);
        if (scalars.size() == 0) {
            return functional::minimum(std::move(tensors));
        } 
        Scalar max_s = functional::minimum(std::move(scalars));
        return functional::minimum(std::move(tensors), max_s);
    }else{
        std::vector<Scalar> scalars =
            functional::min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);
        return functional::minimum(std::move(scalars)); 
    }
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max)
    ntarg_(input),
    ntarg_(dim) = nullptr,
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max, 3, functional_details::max);


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(min)
    ntarg_(input),
    ntarg_(dim) = nullptr,
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(min, 3, functional_details::min);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(argmax)
    ntarg_(input),
    ntarg_(dim) = std::nullopt,
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::argmax);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(argmin)
    ntarg_(input),
    ntarg_(dim) = std::nullopt,
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::argmin);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_indices)
    ntarg_(input),
    ntarg_(dim) = nullptr,
    ntarg_(indices) = nullptr
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_indices, 3, functional_details::max_indices);


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(min_indices)
    ntarg_(input),
    ntarg_(dim) = nullptr,
    ntarg_(indices) = nullptr
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(min_indices, 3, functional_details::min_indices);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(clamp)
    ntarg_(input),
    ntarg_(min) = std::nullopt,
    ntarg_(max) = std::nullopt
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::clamp);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(clamp_)
    ntarg_(input),
    ntarg_(min) = std::nullopt,
    ntarg_(max) = std::nullopt
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::clamp_);

//normalize.h

// Initialization
// This is used to initialize parameters for Layers
namespace init{
namespace functional_details{
NT_ALWAYS_INLINE Tensor& xavier_uniform_(Tensor& input){
    ::nt::functional::xavier_uniform_(input);
    return input;
}

NT_ALWAYS_INLINE TensorGrad& xavier_uniform_(TensorGrad& input){
    ::nt::functional::xavier_uniform_(input.detach());
    return input;
}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(xavier_uniform_)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::xavier_uniform_);

}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(var)
    ntarg_(input),
    ntarg_(dim) = nullptr,
    ntarg_(correction) = 1,
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::var)

//numpy.h 

namespace functional_details{
NT_ALWAYS_INLINE void to_numpy(const Tensor& input, std::string str){::nt::functional::to_numpy(input, str);}
NT_ALWAYS_INLINE void to_numpy(const TensorGrad& input, std::string str){::nt::functional::to_numpy(input.detach(), str);}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(to_numpy)
    ntarg_(input),
    ntarg_(str)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::to_numpy);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(from_numpy)
    ntarg_(str)
NT_FINISH_NAMED_PARAMETER_FUNCTION_(functional::from_numpy);


//operators.h

#define NT_MAKE_OPERATOR_NAME(name, underscore)\
NT_MAKE_NAMED_PARAMETER_FUNCTION_(name)\
    ntarg_(input),\
    ntarg_(other)\
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::name)\
\
NT_MAKE_NAMED_PARAMETER_FUNCTION_(underscore)\
    ntarg_(input),\
    ntarg_(other)\
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::underscore)\

NT_MAKE_OPERATOR_NAME(add, add_);
NT_MAKE_OPERATOR_NAME(multiply, multiply_);
NT_MAKE_OPERATOR_NAME(subtract, subtract_);
NT_MAKE_OPERATOR_NAME(divide, divide_);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(remainder)
    ntarg_(input),
    ntarg_(other)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::remainder);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(fmod)
    ntarg_(input),
    ntarg_(other)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::fmod);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(inverse)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::inverse);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(inverse_)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::inverse_);


#undef NT_MAKE_OPERATOR_NAME 


namespace functional_details{
NT_ALWAYS_INLINE Tensor pad(const Tensor& input, std::vector<Tensor::size_value_t> padding, std::string mode, Scalar value){
    return functional::pad(input, std::move(padding), mode.c_str(), value);
}
template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
NT_ALWAYS_INLINE Tensor pad(const Tensor& input, std::initializer_list<T> padding, std::string mode, Scalar value){
    std::vector<Tensor::size_value_t> padding_(padding.size());
    auto begin = padding.begin();
    auto end = padding.end();
    auto begin2 = padding_.begin();
    for(;begin != end; ++begin, ++begin2){
        *begin2 = static_cast<Tensor::size_value_t>(*begin);
    }
    return functional::pad(input, std::move(padding_), mode.c_str(), value);
}
NT_ALWAYS_INLINE TensorGrad pad(const TensorGrad& input, std::vector<Tensor::size_value_t> padding, std::string mode, Scalar value){
    return functional::pad(input, std::move(padding), mode.c_str(), value);
}
template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
NT_ALWAYS_INLINE TensorGrad pad(const TensorGrad& input, std::initializer_list<T> padding, std::string mode, Scalar value){
    std::vector<Tensor::size_value_t> padding_(padding.size());
    auto begin = padding.begin();
    auto end = padding.end();
    auto begin2 = padding_.begin();
    for(;begin != end; ++begin, ++begin2){
        *begin2 = static_cast<Tensor::size_value_t>(*begin);
    }
    return functional::pad(input, std::move(padding_), mode.c_str(), value);
}

NT_ALWAYS_INLINE Tensor unpad(const Tensor& input, std::vector<Tensor::size_value_t> padding, bool contig){
    return functional::unpad(input, std::move(padding), contig);
}
template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
NT_ALWAYS_INLINE Tensor unpad(const Tensor& input, std::initializer_list<T> padding, bool contig){
    std::vector<Tensor::size_value_t> padding_(padding.size());
    auto begin = padding.begin();
    auto end = padding.end();
    auto begin2 = padding_.begin();
    for(;begin != end; ++begin, ++begin2){
        *begin2 = static_cast<Tensor::size_value_t>(*begin);
    }
    return functional::unpad(input, std::move(padding_), contig);
}

NT_ALWAYS_INLINE TensorGrad unpad(const TensorGrad& input, std::vector<Tensor::size_value_t> padding, bool contig){
    return functional::unpad(input, std::move(padding), contig);
}

template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
NT_ALWAYS_INLINE TensorGrad unpad(const TensorGrad& input, std::initializer_list<T> padding, bool contig){
    std::vector<Tensor::size_value_t> padding_(padding.size());
    auto begin = padding.begin();
    auto end = padding.end();
    auto begin2 = padding_.begin();
    for(;begin != end; ++begin, ++begin2){
        *begin2 = static_cast<Tensor::size_value_t>(*begin);
    }
    return functional::unpad(input, std::move(padding_), contig);
}

}


//padding.h
NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(pad)
    ntarg_(input),
    ntarg_(padding),
    ntarg_(mode) = std::string("constant"),
    ntarg_(value) = 0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(pad, 4, functional_details::pad);


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unpad)
    ntarg_(input),
    ntarg_(padding),
    ntarg_(no_contiguous) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(unpad, 3, functional_details::unpad);



//pooling.h
NT_MAKE_NAMED_PARAMETER_FUNCTION_(avg_pool1d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(ceil_mode) = false,
    ntarg_(count_include_padding) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::avg_pool1d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(adaptive_avg_pool1d)
    ntarg_(input),
    ntarg_(output_size)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::adaptive_avg_pool1d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(avg_pool2d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(ceil_mode) = false,
    ntarg_(count_include_padding) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(avg_pool2d, 6, functional::avg_pool2d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_avg_pool2d)
    ntarg_(input),
    ntarg_(output_size)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_avg_pool2d, 2, functional::adaptive_avg_pool2d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(avg_pool3d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(ceil_mode) = false,
    ntarg_(count_include_padding) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(avg_pool3d, 6, functional::avg_pool3d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_avg_pool3d)
    ntarg_(input),
    ntarg_(output_size)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_avg_pool3d, 2, functional::adaptive_avg_pool3d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_(lp_pool1d)
    ntarg_(input),
    ntarg_(power),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(ceil_mode) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::lp_pool1d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(adaptive_lp_pool1d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(power)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::adaptive_lp_pool1d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(lp_pool2d)
    ntarg_(input),
    ntarg_(power),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(ceil_mode) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(lp_pool2d, 5, functional::lp_pool2d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_lp_pool2d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(power)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_lp_pool2d, 3, functional::adaptive_lp_pool2d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(lp_pool3d)
    ntarg_(input),
    ntarg_(power),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(ceil_mode) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(lp_pool3d, 5, functional::lp_pool3d);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_lp_pool3d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(power)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_lp_pool3d, 3, functional::adaptive_lp_pool3d)



namespace functional_details{
NT_ALWAYS_INLINE TensorGrad max_pool1d(const TensorGrad& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode, bool return_indices){
    return functional::max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode);
}

NT_ALWAYS_INLINE Tensor max_pool1d(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode, bool return_indices){
    return functional::max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode, false);
}

NT_ALWAYS_INLINE TensorGrad max_pool2d(const TensorGrad& input, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, bool ceil_mode, bool return_indices){
    return functional::max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode);
}

NT_ALWAYS_INLINE Tensor max_pool2d(const Tensor& input, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, bool ceil_mode, bool return_indices){
    return functional::max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, false);
}


NT_ALWAYS_INLINE TensorGrad max_pool3d(const TensorGrad& input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, bool ceil_mode, bool return_indices){
    return functional::max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode);
}

NT_ALWAYS_INLINE Tensor max_pool3d(const Tensor& input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, bool ceil_mode, bool return_indices){
    return functional::max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, false);
}


}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(max_pool1d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(ceil_mode) = false,
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::max_pool1d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(max_unpool1d)
    ntarg_(input),
    ntarg_(indices),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(output_size) = -1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::max_unpool1d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(adaptive_max_pool1d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::adaptive_max_pool1d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_pool2d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(ceil_mode) = false,
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_pool2d, 7, functional_details::max_pool2d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_unpool2d)
    ntarg_(input),
    ntarg_(indices),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(output_size) = -1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_unpool2d, 6, functional::max_unpool2d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_max_pool2d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_max_pool2d, 3, functional::adaptive_max_pool2d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_pool3d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(dilation) = 1,
    ntarg_(ceil_mode) = false,
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_pool3d, 7, functional_details::max_pool3d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_unpool3d)
    ntarg_(input),
    ntarg_(indices),
    ntarg_(kernel_size),
    ntarg_(stride) = -1,
    ntarg_(padding) = 0,
    ntarg_(output_size) = -1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(max_unpool3d, 6, functional::max_unpool3d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_max_pool3d)
    ntarg_(input),
    ntarg_(output_size),
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(adaptive_max_pool3d, 3, functional::adaptive_max_pool3d)


NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fractional_max_pool2d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(output_size) = -1,
    ntarg_(output_ratio) = double(-1.0),
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fractional_max_pool2d, 5, functional::fractional_max_pool2d)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fractional_max_pool3d)
    ntarg_(input),
    ntarg_(kernel_size),
    ntarg_(output_size) = -1,
    ntarg_(output_ratio) = double(-1.0),
    ntarg_(return_indices) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(fractional_max_pool3d, 5, functional::fractional_max_pool3d);



//print 

inline void print(const Tensor& t){
    functional::print(t);
}
inline void print(const TensorGrad& tg){
    std::cout << tg << std::endl;
}
inline void print(const Scalar& s){
    std::cout << s << std::endl;
}

//rand
NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(randint)
    ntarg_(low),
    ntarg_(high),
    ntarg_(size),
    ntarg_(dtype) = DType::int64
NT_FINISH_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(randint, 4, functional::randint)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(rand)
    ntarg_(low),
    ntarg_(high),
    ntarg_(size),
    ntarg_(dtype) = DType::Float32
NT_FINISH_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(rand, 4, functional::rand)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(randbools)
    ntarg_(size),
    ntarg_(ratio)
NT_FINISH_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(randbools, 2, functional::randbools)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(randn)
    ntarg_(size),
    ntarg_(dtype) = DType::Float32
NT_FINISH_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(randn, 2, functional::randn)



//for the ranges header:
//  Considering not including them in the nt namespace because:
//      - They are more internal-use [except maybe get_range]
//          - If get_range is upgraded to be user-friendly and not strictly internal
//          - use, maybe that will be used, but for now, it is really only a utility function
//          - that still would need a clear definition and some extra use added [idx used improperly]
//      - The ranges operations are pretty throughly and more easily used by doing the following:
//          nt::Tensor t = nt::rand(low = -4, high = 20, size = {3, 5, 2, 8});
//          nt::Tensor ranged = t(1 <range, 1 <range> 4, 1);
//      - Which is a lot more user friendly
//      - This operation has also already been tested


//repeat
namespace functional_details{
NT_ALWAYS_INLINE Tensor repeat(const Tensor& input, int64_t amt, int64_t dim){
    if(dim == -1){
        return ::nt::functional::repeat_(input, amt);
    }else{
        return ::nt::functional::repeat_(input, dim, amt);
    }
}

NT_ALWAYS_INLINE TensorGrad repeat(const TensorGrad& input, int64_t amt, int64_t dim){
    if(dim == -1){
        return ::nt::functional::repeat_(input, amt);
    }else{
        return ::nt::functional::repeat_(input, dim, amt);
    }
}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(repeat)
    ntarg_(input),
    ntarg_(amt),
    ntarg_(dim) = -1
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::repeat)



NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(expand)
    ntarg_(input),
    ntarg_(size)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(expand, 2, functional::expand)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(expand_as)
    ntarg_(input),
    ntarg_(other)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::expand_as)

//round.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_(round)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::round)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(floor)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::floor)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(ceil)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::ceil)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(trunc)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::trunc)


// save_load.h

namespace functional_details{
NT_ALWAYS_INLINE void save(const Tensor& input, const std::string& str){::nt::functional::save(input, str.c_str());}
NT_ALWAYS_INLINE void save(const TensorGrad& input, const std::string& str){::nt::functional::save(input.detach(), str.c_str());}
NT_ALWAYS_INLINE void load(const std::string& str){::nt::functional::load(str.c_str());}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(save)
    ntarg_(input),
    ntarg_(str)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::save)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(load)
    ntarg_(str)
NT_FINISH_NAMED_PARAMETER_FUNCTION_(functional_details::load);


//softmax.h
namespace functional_details{
NT_ALWAYS_INLINE Tensor softmax(const Tensor& input, std::optional<int64_t> dim, bool stable){
    if(stable){
        if(dim.has_value()){
            return ::nt::functional::softmax_stable(input, dim.value());
        }
        return ::nt::functional::softmax_stable(input);
    }
    return dim.has_value() ? ::nt::functional::softmax(input, dim.value()) : ::nt::functional::softmax(input);
}

NT_ALWAYS_INLINE TensorGrad softmax(const TensorGrad& input, std::optional<int64_t> dim, bool stable){
    return functional::softmax(input, dim, stable);
}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_(softmax)
    ntarg_(input),
    ntarg_(dim) = std::nullopt,
    ntarg_(stable) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional_details::softmax)

//inline TensorGrad gumbel_softmax(const TensorGrad& input, Scalar tau, bool hard, int64_t dim = -1, bool stable = true)
NT_MAKE_NAMED_PARAMETER_FUNCTION_(gumbel_softmax)
    ntarg_(input),
    ntarg_(tau) = nt::complex_64(1, 1),
    ntarg_(hard) = false,
    ntarg_(dim) = -1,
    ntarg_(stable) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::gumbel_softmax)


//sort.h <NEED TO MAKE TESTS PAST THIS POINT>
NT_MAKE_NAMED_PARAMETER_FUNCTION_(sort)
    ntarg_(input),
    ntarg_(dim) = -1,
    ntarg_(descending) = false,
    ntarg_(return_sorted) = true,
    ntarg_(return_indices) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::sort)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(coordsort)
    ntarg_(input),
    ntarg_(dim) = -2,
    ntarg_(descending) = false,
    ntarg_(return_sorted) = true,
    ntarg_(return_indices) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::coordsort)


// split.h

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(split)
    ntarg_(input),
    ntarg_(dim),
    ntarg_(splitting) = nullptr
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(split, 3, functional::split)


NT_MAKE_NAMED_PARAMETER_FUNCTION_(chunk)
    ntarg_(input),
    ntarg_(chunks),
    ntarg_(dim) = 0
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::chunk)

//stride.h
NT_MAKE_NAMED_PARAMETER_FUNCTION_(diagonal)
    ntarg_(input),
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::diagonal)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(as_strided)
    ntarg_(input),
    ntarg_(size),
    ntarg_(stride),
    ntarg_(storage_offset) = 0,
    ntarg_(whole_tensor) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(as_strided, 5, functional::as_strided);



//sum_exp_log.h
NT_MAKE_NAMED_PARAMETER_FUNCTION_(log)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::log)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(exp)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::exp)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(log_)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::log_)

NT_MAKE_NAMED_PARAMETER_FUNCTION_(exp_)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::exp_)

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(sum)
    ntarg_(input),
    ntarg_(dim) = nullptr,
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(sum, 3, functional::sum);

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(logsumexp)
    ntarg_(input),
    ntarg_(dim) = nullptr,
    ntarg_(keepdim) = false
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(logsumexp, 3, functional::logsumexp);

//transpose.h

namespace functional_details{
NT_ALWAYS_INLINE Tensor permute(const Tensor& input, std::vector<Tensor::size_value_t> permutations){
    return functional::permute(input, std::move(permutations));
}
template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
NT_ALWAYS_INLINE Tensor permute(const Tensor& input, std::initializer_list<T> permutations){
    std::vector<Tensor::size_value_t> permutations_(permutations.size());
    auto begin = permutations.begin();
    auto end = permutations.end();
    auto begin2 = permutations_.begin();
    for(;begin != end; ++begin, ++begin2){
        *begin2 = static_cast<Tensor::size_value_t>(*begin);
    }
    return functional::permute(input, std::move(permutations_));
}

NT_ALWAYS_INLINE TensorGrad permute(const TensorGrad& input, std::vector<Tensor::size_value_t> permutations){
    return functional::permute(input, std::move(permutations));
}
template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
NT_ALWAYS_INLINE TensorGrad permute(const TensorGrad& input, std::initializer_list<T> permutations){
    std::vector<Tensor::size_value_t> permutations_(permutations.size());
    auto begin = permutations.begin();
    auto end = permutations.end();
    auto begin2 = permutations_.begin();
    for(;begin != end; ++begin, ++begin2){
        *begin2 = static_cast<Tensor::size_value_t>(*begin);
    }
    return functional::permute(input, std::move(permutations_));
}
}

NT_MAKE_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(permute)
    ntarg_(input),
    ntarg_(dims) 
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_WITH_INIT_LIST_(permute, 2, functional_details::permute);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(transpose)
    ntarg_(input),
    ntarg_(dim0), 
    ntarg_(dim1) 
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::transpose);


NT_MAKE_NAMED_PARAMETER_FUNCTION_(row_col_swap_)
    ntarg_(input)
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::row_col_swap_);

//trig.h



NT_MAKE_NAMED_ACTIVATION_FUNCTION(tan);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(tanh);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(atan);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(atanh);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(cotan);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(cotanh);

NT_MAKE_NAMED_ACTIVATION_FUNCTION(sin);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(sinh);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(asin);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(asinh);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(csc);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(csch);

NT_MAKE_NAMED_ACTIVATION_FUNCTION(cos);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(cosh);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(acos);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(acosh);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(sec);
NT_MAKE_NAMED_ACTIVATION_FUNCTION(sech);

NT_MAKE_NAMED_PARAMETER_FUNCTION_(unique)
    ntarg_(input),
    ntarg_(dim) = std::nullopt,
    ntarg_(return_unique) = true,
    ntarg_(return_indices) = true
NT_OVERLOAD_NAMED_PARAMETER_FUNCTION_(functional::unique)


#undef NT_MAKE_NAMED_ACTIVATION_FUNCTION 
#undef ADD_UNDERSCORE 
#undef NT_EXPAND
#undef NT_LARG
#undef NT_TARG

}


#ifdef NT_DEFINE_PARAMETER_ARGUMENTS
#define NT_DEFINE_ARGUMENT(name) inline static constexpr auto name = ntarg_(name);
namespace nt::literals{
NT_DEFINE_ARGUMENT(amt)
NT_DEFINE_ARGUMENT(atol)
NT_DEFINE_ARGUMENT(beta)
NT_DEFINE_ARGUMENT(bias)
NT_DEFINE_ARGUMENT(ceil_mode)
NT_DEFINE_ARGUMENT(count_include_padding)
NT_DEFINE_ARGUMENT(correction)
NT_DEFINE_ARGUMENT(chunks)
NT_DEFINE_ARGUMENT(descending)
NT_DEFINE_ARGUMENT(dilation)
NT_DEFINE_ARGUMENT(dim)
NT_DEFINE_ARGUMENT(dims)
NT_DEFINE_ARGUMENT(dim0)
NT_DEFINE_ARGUMENT(dim1)
NT_DEFINE_ARGUMENT(dtype)
NT_DEFINE_ARGUMENT(equal_nan)
NT_DEFINE_ARGUMENT(exponent)
NT_DEFINE_ARGUMENT(groups)
NT_DEFINE_ARGUMENT(hard)
NT_DEFINE_ARGUMENT(high)
NT_DEFINE_ARGUMENT(idx)
NT_DEFINE_ARGUMENT(image)
NT_DEFINE_ARGUMENT(indices)
NT_DEFINE_ARGUMENT(input)
NT_DEFINE_ARGUMENT(keepdim)
NT_DEFINE_ARGUMENT(kernel)
NT_DEFINE_ARGUMENT(kernel_size)
NT_DEFINE_ARGUMENT(list)
NT_DEFINE_ARGUMENT(low)
NT_DEFINE_ARGUMENT(max)
NT_DEFINE_ARGUMENT(min)
NT_DEFINE_ARGUMENT(mode)
NT_DEFINE_ARGUMENT(name)
NT_DEFINE_ARGUMENT(no_contiguous)
NT_DEFINE_ARGUMENT(num)
NT_DEFINE_ARGUMENT(num_classes)
NT_DEFINE_ARGUMENT(other)
NT_DEFINE_ARGUMENT(output)
NT_DEFINE_ARGUMENT(output_padding)
NT_DEFINE_ARGUMENT(output_size)
NT_DEFINE_ARGUMENT(output_ratio)
NT_DEFINE_ARGUMENT(padding)
NT_DEFINE_ARGUMENT(power)
NT_DEFINE_ARGUMENT(r)
NT_DEFINE_ARGUMENT(ratio)
NT_DEFINE_ARGUMENT(return_indices)
NT_DEFINE_ARGUMENT(return_sorted)
NT_DEFINE_ARGUMENT(return_unique)
NT_DEFINE_ARGUMENT(rtol)
NT_DEFINE_ARGUMENT(size)
NT_DEFINE_ARGUMENT(splitting)
NT_DEFINE_ARGUMENT(start)
NT_DEFINE_ARGUMENT(storage_offset)
NT_DEFINE_ARGUMENT(str)
NT_DEFINE_ARGUMENT(stride)
NT_DEFINE_ARGUMENT(tau)
NT_DEFINE_ARGUMENT(tensor)
NT_DEFINE_ARGUMENT(tensor1)
NT_DEFINE_ARGUMENT(tensor2)
NT_DEFINE_ARGUMENT(tensors)
NT_DEFINE_ARGUMENT(threshold)
NT_DEFINE_ARGUMENT(transpose_a)
NT_DEFINE_ARGUMENT(transpose_b)
NT_DEFINE_ARGUMENT(transpose_out)
NT_DEFINE_ARGUMENT(type)
NT_DEFINE_ARGUMENT(val)
NT_DEFINE_ARGUMENT(value)
NT_DEFINE_ARGUMENT(vec)
NT_DEFINE_ARGUMENT(weight)
NT_DEFINE_ARGUMENT(whole_tensor)

}
#undef NT_DEFINE_ARGUMENT
#endif //NT_DEFINE_PARAMETER_ARGUMENTS 

#endif
