#include "../functional/functional.h"
#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "TensorGrad.h"
#include "functional_class.h"
#include "../dtype/ArrayVoid.hpp"
// #include "functional.h"

namespace nt {
namespace functional {
TensorGrad TensorGrad_Functional_Class::matmult(const TensorGrad &a, const TensorGrad &b) {
    if (!a.do_track_grad) {
        if (!b.do_track_grad) {
            Tensor out = ::nt::functional::matmult(a.tensor, b.tensor);
            TensorGrad result(std::move(out), false);
            result.do_track_grad = false;
            return std::move(result);
        }
        return matmult(a.tensor, b);
    }
    if (!b.do_track_grad) {
        return matmult(a, b.tensor);
    }
    // a and b are going to have to be cloned anyways so:

    intrusive_ptr<tensor_holder> a_c =
            make_intrusive<tensor_holder>(a.tensor.clone());
    intrusive_ptr<tensor_holder> b_c =
            make_intrusive<tensor_holder>(b.tensor.clone());
    TensorGrad result(::nt::functional::matmult(a_c->tensor, b_c->tensor), a.grad_required);
    result.track_tensors(a, b);

    // Define backward function
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
                parents[0]->grad->tensor +=
                        ::nt::functional::matmult(grad, b->tensor, false, true);

                parents[1]->grad->tensor +=
                        ::nt::functional::matmult(a->tensor, grad, true, false);
            },
            a_c, b_c);
    return result;
}

TensorGrad TensorGrad_Functional_Class::matmult(const Tensor &a, const TensorGrad &b) {
    if (!b.do_track_grad) {
        Tensor out = ::nt::functional::matmult(a, b.tensor);
        TensorGrad result(std::move(out), b.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> a_c = make_intrusive<tensor_holder>(a.clone());
    TensorGrad result(::nt::functional::matmult(a_c->tensor, b.tensor), b.grad_required);
    result.track_tensors(b);

    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> a) {
                parents[0]->grad->tensor +=
                        ::nt::functional::matmult(a->tensor, grad, 1, 0);
            },
            a_c);
    return result;
}

TensorGrad TensorGrad_Functional_Class::matmult(const TensorGrad &a, const Tensor &b) {
    if (!a.do_track_grad) {
        Tensor out = ::nt::functional::matmult(a.tensor, b);
        TensorGrad result(std::move(out), a.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> b_c = make_intrusive<tensor_holder>(b.clone());
    TensorGrad result(::nt::functional::matmult(a.tensor, b_c->tensor), a.grad_required);
    result.track_tensors(a);

    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> b) {
                parents[0]->grad->tensor +=
                        ::nt::functional::matmult(grad, b->tensor, 0, 1);
            },
            b_c);
    return result;
}

TensorGrad TensorGrad_Functional_Class::unfold3d(
        const TensorGrad &x, utils::my_n_tuple<3> kernel_size,
        utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding,
        utils::my_n_tuple<3> stride, bool transpose_out) {
    TensorGrad result(::nt::functional::unfold3d(x.tensor, kernel_size, dilation, padding, stride,
                                                         transpose_out), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                utils::my_n_tuple<3> output_size(parents[0]->grad->tensor.shape()[-3],
                                                                                 parents[0]->grad->tensor.shape()[-2],
                                                                                 parents[0]->grad->tensor.shape()[-1]);
                ::nt::functional::unfold3d_backward(grad, parents[0]->grad->tensor, output_size,
                                                    kernel_size, dilation, padding, stride,
                                                    transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::unfold1d(
        const TensorGrad &x, Tensor::size_value_t kernel_size,
        Tensor::size_value_t dilation, Tensor::size_value_t padding,
        Tensor::size_value_t stride, bool transpose_out) {
    TensorGrad result(::nt::functional::unfold1d(x.tensor, kernel_size, dilation, padding, stride,
                                                         transpose_out), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                Tensor::size_value_t output_size = parents[0]->grad->tensor.shape()[-1];
                ::nt::functional::unfold1d_backward(grad, parents[0]->grad->tensor, output_size,
                                                    kernel_size, dilation, padding, stride,
                                                    transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fold(const TensorGrad &x,
                                             utils::my_tuple output_size,
                                             utils::my_tuple kernel_size,
                                             utils::my_tuple dilation,
                                             utils::my_tuple padding,
                                             utils::my_tuple stride) {
    TensorGrad result(
            ::nt::functional::fold(x.tensor, output_size, kernel_size, dilation, padding, stride), x.grad_required);
    result.track_tensors(x);
    // it is coppied because the backward pass will go out of scope of this
    // function and so I dont want that memory to try to be referenced
    result.create_backward_function(
            [output_size, kernel_size, dilation, padding, stride](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                ::nt::functional::fold_backward(grad, parents[0]->grad->tensor, output_size, kernel_size,
                                            dilation, padding, stride);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::unfold(
        const TensorGrad &x, utils::my_tuple kernel_size, utils::my_tuple dilation,
        utils::my_tuple padding, utils::my_tuple stride, bool transpose_out) {
    TensorGrad result(
            ::nt::functional::unfold(x.tensor, kernel_size, dilation, padding, stride, transpose_out), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [kernel_size, dilation, padding, stride, transpose_out](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents) {
                utils::my_tuple output_size(parents[0]->grad->tensor.shape()[-2],
                                                                        parents[0]->grad->tensor.shape()[-1]);
                ::nt::functional::unfold_backward(grad, parents[0]->grad->tensor, output_size,
                                                kernel_size, dilation, padding, stride, transpose_out);
            });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::sigmoid(const TensorGrad &x) {
    Tensor a = ::nt::functional::sigmoid(x.tensor);
    intrusive_ptr<tensor_holder> sigmoid_x =
            make_intrusive<tensor_holder>(a.clone());
    TensorGrad result(std::move(a), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> x) {
                parents[0]->grad->tensor += grad * ::nt::functional::dsigmoid(x->tensor, false);
            },
            sigmoid_x);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::silu(const TensorGrad &x) {
    Tensor a = ::nt::functional::silu(x.tensor);
    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.tensor.clone());
    TensorGrad result(std::move(a), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> x) {
                parents[0]->grad->tensor += grad * ::nt::functional::dsilu(x->tensor);
            },
            saved_x);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::gelu(const TensorGrad &x) {
    // Forward pass
    Tensor a = ::nt::functional::gelu(x.tensor);

    // Save the input tensor for use in the backward pass
    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.tensor.clone());

    // Create TensorGrad object for the result
    TensorGrad result(std::move(a), x.grad_required);
    result.track_tensors(x);

    // Define the backward function
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                // Compute the gradient using the saved input tensor
                parents[0]->grad->tensor += grad * ::nt::functional::dgelu(saved_x->tensor);
            },
            saved_x);

    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::clamp(const TensorGrad &x,
                                                std::optional<int64_t> min,
                                                std::optional<int64_t> max) {
    TensorGrad out = x.clone();
    if (min && max) {
        out[out < min.value() && out > max.value()] = 0;
        return std::move(out);
    } else if (min)
        out[out < min.value()] = 0;
    else if (max)
        out[out > max.value()] = 0;
    return std::move(out);
}

TensorGrad TensorGrad_Functional_Class::relu(const TensorGrad &x) {
    return clamp(x, 0, std::nullopt);
}

TensorGrad TensorGrad_Functional_Class::var(const TensorGrad &x,
                                            utils::optional_list dim,
                                            int64_t correction, bool keepdim) {
    if (!x.do_track_grad) {
        Tensor out = ::nt::functional::var(x.tensor, dim, correction, keepdim);
        TensorGrad result(std::move(out), x.grad_required);
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> x_c =
            make_intrusive<tensor_holder>(x.tensor.clone());
    TensorGrad result(
            ::nt::functional::var(x_c->tensor, dim, correction, keepdim), x.grad_required);
    result.track_tensors(x);
    result.create_backward_function(
            [dim, correction](const Tensor &grad,
                                                std::vector<intrusive_ptr<TensorGrad>> &parents,
                                                intrusive_ptr<tensor_holder> x) {
                parents[0]->grad->tensor += ::nt::functional::dvar(grad, x->tensor, dim, correction);
            },
            x_c);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::invsqrt(const TensorGrad &x) {
    TensorGrad result(::nt::functional::invsqrt(x.tensor), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }

    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                                                    intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->grad->tensor += ::nt::functional::dinvsqrt(saved_x->tensor);
            },
            make_intrusive<tensor_holder>(x.tensor.clone()));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::sqrt(const TensorGrad &x) {
    TensorGrad result(::nt::functional::sqrt(x.tensor), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }

    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                                                    intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->grad->tensor += ::nt::functional::dsqrt(saved_x->tensor);
            },
            make_intrusive<tensor_holder>(x.tensor.clone()));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::tanh(const TensorGrad &x) {
    TensorGrad result(::nt::functional::tanh(x.tensor), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }

    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->grad->tensor += ::nt::functional::dtanh(saved_x->tensor);
            },
            make_intrusive<tensor_holder>(x.tensor.clone()));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::tan(const TensorGrad &x) {
    TensorGrad result(::nt::functional::tan(x.tensor), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }

    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->grad->tensor += ::nt::functional::dtan(saved_x->tensor);
            },
            make_intrusive<tensor_holder>(x.tensor.clone()));
    return std::move(result);
}

Tensor cat_vec(std::vector<TensorGrad> &tgs) {
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = begin->shape();
    const SizeRef sh_smaller = sh.pop_front();
    int64_t n_dim_size = sh[0];
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += begin->shape()[0];
        utils::THROW_EXCEPTION(begin->shape().pop_front() == sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     begin->shape().pop_front(), sh_smaller);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<std::reference_wrapper<const ArrayVoid>> arrVds;
    arrVds.reserve(num); // okay because it is allocating a reference wrapper,
                                             // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(std::cref(begin->tensor.arr_void()));
    }
    return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
}

Tensor cat_vec(std::vector<TensorGrad> &tgs, int64_t dim) {

    if (dim == 0) {
        return cat_vec(tgs);
    }
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = begin->shape().transpose(0, dim);
    int64_t n_dim_size = sh[0];
    const SizeRef sh_smaller = sh.pop_front();
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += begin->shape()[dim];
        utils::THROW_EXCEPTION(begin->shape().transpose(0, dim).pop_front() ==
                                                             sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     begin->shape(), sh);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<ArrayVoid> arrVds;
    //arrVds.reserve(num); // okay because it is allocating a reference wrapper,
    // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(begin->tensor.transpose(0, dim).arr_void());
    }
    SizeRef shape(std::move(vec));
    return Tensor(ArrayVoid::cat(arrVds), std::move(shape)).transpose(0, dim);
}

Tensor cat_vec_grad(std::vector<intrusive_ptr<TensorGrad>> &tgs) {
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = (*begin)->shape();
    const SizeRef sh_smaller = sh.pop_front();
    int64_t n_dim_size = sh[0];
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += (*begin)->shape()[0];
        utils::THROW_EXCEPTION((*begin)->shape().pop_front() == sh_smaller,
                                                     "Expected all shapes in concatenate to be the "
                                                     "same, but got $ and $",
                                                     (*begin)->shape().pop_front(), sh_smaller);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<std::reference_wrapper<const ArrayVoid>> arrVds;
    arrVds.reserve(num); // okay because it is allocating a reference wrapper,
                                             // putting a number there would cause an allocation error
    begin = begin_cpy;
    typename SizeRef::value_type i = 0;
    for (typename SizeRef::value_type i = 0; begin != end; ++begin, ++i) {
        arrVds.push_back(std::cref((*begin)->grad->tensor.arr_void()));
    }
    return Tensor(ArrayVoid::cat(arrVds), SizeRef(std::move(vec)));
}

Tensor cat_vec_grad(std::vector<intrusive_ptr<TensorGrad>> &tgs, int64_t dim) {
    if (dim == 0) {
        return cat_vec_grad(tgs);
    }
    const typename SizeRef::value_type &num = tgs.size();
    auto begin = tgs.begin();
    auto end = tgs.end();
    const SizeRef sh = (*begin)->shape().transpose(0, dim);
    int64_t n_dim_size = sh[0];
    const SizeRef sh_smaller = sh.pop_front();
    auto begin_cpy = begin;
    ++begin;
    for (; begin != end; ++begin) {
        n_dim_size += (*begin)->shape()[dim];
        utils::THROW_EXCEPTION(
                (*begin)->shape().transpose(0, dim).pop_front() == sh_smaller,
                "Expected all shapes in concatenate to be the same, but got $ and "
                "$",
                (*begin)->shape(), sh);
    }
    std::vector<typename SizeRef::value_type> vec = sh.Vec();
    vec[0] = n_dim_size;
    std::vector<ArrayVoid> arrVds;
    //arrVds.reserve(num); // okay because it is allocating a reference wrapper,
                                             // putting a number there would cause an allocation error
    begin = begin_cpy;
    // typename SizeRef::value_type i = 0;
    for (;begin != end; ++begin) {
        arrVds.push_back((*begin)->grad->tensor.transpose(0, dim).arr_void());
    }
    SizeRef shape(std::move(vec));
    return Tensor(ArrayVoid::cat(arrVds), std::move(shape)).transpose(0, dim);
}

TensorGrad TensorGrad_Functional_Class::cat(std::vector<TensorGrad> tgs, int64_t dim) {
    bool track_grad = tgs[0].do_track_grad;
    bool require_grad = tgs[0].grad_required;
    for (const auto &tg : tgs) {
        utils::throw_exception(tg.do_track_grad == track_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(tg.grad_required == require_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(!tg.is_null(), "Cannot concatenate null tensors");
    }
    if (!require_grad) {
        track_grad = false;
    }
    TensorGrad result(cat_vec(tgs, dim), require_grad);
    if (!track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }

    // tracking the gradient itself
    // rather than tracking each parent individually
    for (const auto &tg : tgs) {
        if (tg.grad == nullptr) {
            tg.grad =
                    make_intrusive<tensor_holder>(::nt::functional::zeros_like(tg.tensor));
        }
    }
    result.track_tensors(tgs);
    result.grad = make_intrusive<tensor_holder>(cat_vec_grad(result.parents->get(), dim));
    return std::move(result);
}

// inline std::vector<Tensor> vectorize(Tensor& t){
//     utils::throw_exception(t.dtype == DType::TensorObj,
//                            "can only vectorize tensor of tensors");
//     return t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj > > >
//         ([](auto begin, auto end) -> std::vector<Tensor> {return std::vector<Tensor>(begin, end);});

// }

TensorGrad TensorGrad_Functional_Class::cat(TensorGrad tgs, int64_t dim) {
    // std::cout << tgs << std::endl;
    // std::cout << "on dim "<<dim<<std::endl;
    // std::vector<std::reference_wrapper<Tensor>> first_cat;
    // first_cat.reserve(tgs.numel());
    // Tensor* begin = reinterpret_cast<Tensor*>(tgs.tensor.data_ptr());
    // Tensor* end = begin + tgs.numel();
    // for(;begin != end; ++begin)
    //     first_cat.push_back(std::ref(*begin));

    // for(int64_t i = 0; i < tgs.numel(); ++i)
    //     first_cat[i] = tgs[i].tensor;
    TensorGrad result(::nt::functional::cat(tgs.tensor, dim), tgs.grad_required);

    // if(tgs.grad == nullptr){
    //     std::cout << tgs.tensor.dtype << std::endl;
    //     Tensor zeros = ::nt::functional::zeros_like(tgs.tensor);
    //     std::cout << "zeros: "<<zeros<<std::endl;
    // }
    // else{
    //     std::cout << "tgs.grad is not nullptr"<<std::endl;
    //     std::cout << tgs.grad->tensor << std::endl;
    //     std::cout << ::nt::functional::zeros_like(tgs.tensor);
    // }
    result.track_grad(tgs, [dim](Tensor &grad) {return ::nt::functional::cat(grad, dim); });
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::stack(std::vector<TensorGrad> tgs, int64_t dim) {
    bool track_grad = tgs[0].do_track_grad;
    bool require_grad = tgs[0].grad_required;
    for (const auto &tg : tgs) {
        utils::throw_exception(tg.do_track_grad == track_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(tg.grad_required == require_grad,
                                                     "Cannot concatenate tensors that are both tracking "
                                                     "the gradient and are not");
        utils::throw_exception(!tg.is_null(), "Cannot concatenate null tensors");
    }
    if (!require_grad) {
        track_grad = false;
    }
    std::vector<std::reference_wrapper<Tensor>> tgs_data_ref;
    tgs_data_ref.reserve(tgs.size());
    for (int64_t i = 0; i < tgs.size(); ++i) {
        tgs_data_ref.push_back(std::ref(tgs[i].tensor));
    }

    TensorGrad result(::nt::functional::stack(tgs_data_ref, dim), require_grad);
    if (!track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }
    std::vector<std::reference_wrapper<Tensor>> tgs_grad_ref;
    tgs_grad_ref.reserve(tgs.size());
    for (const auto &tg : tgs) {
        if (tg.grad == nullptr) {
            tg.grad =
                    make_intrusive<tensor_holder>(functional::zeros_like(tg.tensor));
        }
        // result.parents.push_back(make_intrusive<TensorGrad>(tg));
        tgs_grad_ref.push_back(std::ref(tg.grad->tensor));
    }
    result.grad = make_intrusive<tensor_holder>(::nt::functional::stack(tgs_grad_ref, dim));
    result.track_tensors(tgs);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::stack(TensorGrad tgs, int64_t dim) {
    TensorGrad result(::nt::functional::stack(tgs.tensor, dim), tgs.grad_required);
    result.track_grad(tgs, [dim](Tensor &grad) { return ::nt::functional::stack(grad, dim); });
    return std::move(result);
}



TensorGrad TensorGrad_Functional_Class::split(
        TensorGrad input, std::vector<typename Tensor::size_value_t> splits, int64_t dim) {
    TensorGrad result(::nt::functional::split(input.tensor, splits, dim), input.grad_required);
    result.track_grad(
            input, [splits, dim](Tensor &grad) { return ::nt::functional::split(grad, splits, dim); });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::split(
        TensorGrad input, typename Tensor::size_value_t splits, int64_t dim) {
    TensorGrad result(::nt::functional::split(input.tensor, splits, dim), input.grad_required);
    result.track_grad(
            input, [splits, dim](Tensor &grad) { return ::nt::functional::split(grad, splits, dim); });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::chunk(
        TensorGrad input, typename Tensor::size_value_t chunks, int64_t dim) {
    TensorGrad result(::nt::functional::chunk(input.tensor, chunks, dim), input.grad_required);
    result.track_grad(
            input, [chunks, dim](Tensor &grad) { return ::nt::functional::chunk(grad, chunks, dim); });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::log(const TensorGrad &x) {
    TensorGrad result(::nt::functional::log(x.tensor), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }

    result.track_tensors(x);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_x) {
                parents[0]->grad->tensor += ::nt::functional::dlog(saved_x->tensor);
            },
            make_intrusive<tensor_holder>(x.tensor.clone()));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::softplus(const TensorGrad &x,
                                                     Scalar beta,
                                                     Scalar threshold) {
    Tensor softplus_x = x.tensor * beta;

    Tensor where = softplus_x < threshold;
    if (!::nt::functional::any(where)) {
        return x;
    }

    softplus_x[where].set_(::nt::functional::log(1 + std::exp(softplus_x[where])).divide_(beta));
    TensorGrad result(std::move(softplus_x), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> sx_c = make_intrusive<tensor_holder>(x.tensor.clone());
    intrusive_ptr<tensor_holder> wx_c = make_intrusive<tensor_holder>(where);

    result.track_tensors(x);
    result.create_backward_function(
            [beta](
                    const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                    intrusive_ptr<tensor_holder> x, intrusive_ptr<tensor_holder> where) {
                Tensor x_w = x->tensor[where];
                Tensor grad_w = grad[where];
                x_w *= -beta;
                x_w.exp_();
                x_w += 1;
                x_w.inverse_();
                grad_w *= x_w;
                parents[0]->grad->tensor += grad;
            },
            sx_c, wx_c);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::dropout(const TensorGrad &inputs, double p) {
    Tensor bools = ::nt::functional::randbools(inputs.shape(), p);
    Tensor out = inputs.tensor.clone();
    out[bools] = 0;
    TensorGrad result(out, inputs.grad_required);
    if (!inputs.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }
    result.track_tensors(inputs);
    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
                 intrusive_ptr<tensor_holder> saved_bools) {
                parents[0]->grad->tensor += grad;
                parents[0]->grad->tensor[saved_bools->tensor] = 0;
            },
            make_intrusive<tensor_holder>(bools));
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::abs(const TensorGrad &x){
    Tensor a = ::nt::functional::abs(x.tensor);
    TensorGrad result(std::move(a), x.grad_required);
    if (!x.do_track_grad) {
        result.do_track_grad = false;
        return std::move(result);
    }


    intrusive_ptr<tensor_holder> saved_x =
            make_intrusive<tensor_holder>(x.tensor.clone());
    
    result.track_tensors(x);


    result.create_backward_function(
            [](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents,
               intrusive_ptr<tensor_holder> saved_x) {
                //compute the gradient using the saved input tensor
                Tensor sign_grad = (saved_x->tensor > 0).to(DType::Float32) -
                                   (saved_x->tensor < 0).to(DType::Float32); //compute sign
                parents[0]->grad->tensor += grad * sign_grad;
            },
            saved_x);

    return std::move(result);


}




TensorGrad  TensorGrad_Functional_Class::conv1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv1d(image, kernel.tensor, stride, padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image, kernel.tensor, stride, padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad->tensor, {image_shape[-1]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        TensorGrad result(::nt::functional::conv1d(image.tensor, kernel, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image.tensor, kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {dilation},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv1d(image.tensor, kernel, stride, padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv1d(image, kernel.tensor, stride, padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv1d(image.tensor, kernel.tensor, stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {dilation},
                                     groups);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad->tensor, {image_shape[-1]}, groups);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if(kernel.grad_required == false || kernel.do_track_grad == false){
        //if the kernel isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv2d(image, kernel.tensor, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image, kernel.tensor, stride, padding, dilation, groups, original_x));
    result.track_tensors( kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
          ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad->tensor, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false){
        //if one of the tensors isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv2d(image.tensor, kernel, stride, padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image.tensor, kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
    }, original_w);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv2d(image.tensor, kernel, stride, padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv2d(image, kernel.tensor, stride, padding, dilation, groups);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv2d(image.tensor, kernel.tensor, stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad->tensor, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}



TensorGrad  TensorGrad_Functional_Class::conv3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv3d(image, kernel.tensor, stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image, kernel.tensor, stride, padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[0]->grad->tensor, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);

}

TensorGrad  TensorGrad_Functional_Class::conv3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(image.grad_required == false || image.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv3d(image.tensor, kernel, stride, padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image.tensor, kernel, stride, padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv3d(image.tensor, kernel, stride, padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv3d(image, kernel.tensor, stride, padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv3d(image.tensor, kernel.tensor, stride, padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::conv_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
        ::nt::functional::conv_dkernel(grad, img->tensor, parents[1]->grad->tensor, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const Tensor& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv_transpose1d(image, kernel.tensor, stride, padding, output_padding, dilation, groups), false);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose1d(image, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function(
        [image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad->tensor, {padding}, {image_shape[-1]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const TensorGrad& image, const Tensor& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        TensorGrad result(::nt::functional::conv_transpose1d(image.tensor, kernel, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose1d(image.tensor, kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function(
        [image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {output_padding},
                                     {dilation},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose1d(const TensorGrad& image, const TensorGrad& kernel, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv_transpose1d(image.tensor, kernel, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv_transpose1d(image, kernel.tensor, stride, padding, output_padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose1d(image.tensor, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-1]},
                                     {stride},
                                     {padding},
                                     {output_padding},
                                     {dilation},
                                     groups);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad->tensor, {padding}, {image_shape[-1]}, groups);
    }, original_x, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const Tensor& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if(kernel.grad_required == false || kernel.do_track_grad == false){
        //if the kernel isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv_transpose2d(image, kernel.tensor, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose2d(image, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors( kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
          ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad->tensor, {padding[0], padding[1]}, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const TensorGrad& image, const Tensor& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false){
        //if one of the tensors isn't tracking the gradient, then the gradient for neither is tracked
        TensorGrad result(::nt::functional::conv_transpose2d(image.tensor, kernel, stride, padding, output_padding, dilation, groups), false);
        return std::move(result);
    }
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose2d(image.tensor, kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {output_padding[0], output_padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
    }, original_w);
    return std::move(result);
}


TensorGrad  TensorGrad_Functional_Class::conv_transpose2d(const TensorGrad& image, const TensorGrad& kernel, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_padding, utils::my_tuple dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv_transpose2d(image.tensor, kernel, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv_transpose2d(image, kernel.tensor, stride, padding, output_padding, dilation, groups);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose2d(image.tensor, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1]},
                                     {padding[0], padding[1]},
                                     {output_padding[0], output_padding[1]},
                                     {dilation[0], dilation[1]},
                                     groups);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad->tensor, {padding[0], padding[1]}, {image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}



TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const Tensor& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv_transpose3d(image, kernel.tensor, stride, padding, output_padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose3d(image, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x));
    result.track_tensors(kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img){
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[0]->grad->tensor, {padding[0], padding[1], padding[2]}, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x);
    return std::move(result);

}

TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const TensorGrad& image, const Tensor& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
    if(image.grad_required == false || image.do_track_grad == false ){
        return TensorGrad(::nt::functional::conv_transpose3d(image.tensor, kernel, stride, padding, output_padding, dilation, groups), false);
    }

    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose3d(image.tensor, kernel, stride, padding, output_padding, dilation, groups, nullptr, original_w));
    result.track_tensors(image);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {output_padding[0], output_padding[1], output_padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
    }, original_w);
    return std::move(result);
}

TensorGrad  TensorGrad_Functional_Class::conv_transpose3d(const TensorGrad& image, const TensorGrad& kernel, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_padding, utils::my_n_tuple<3> dilation, int64_t groups){
	if(image.grad_required == false || image.do_track_grad == false ){
        return conv_transpose3d(image.tensor, kernel, stride, padding, output_padding, dilation, groups);
    }
    if(kernel.grad_required == false || kernel.do_track_grad == false ){
        return conv_transpose3d(image, kernel.tensor, stride, padding, output_padding, dilation, groups);
    }
    intrusive_ptr<tensor_holder> original_x = make_intrusive<tensor_holder>(Tensor::Null());
    intrusive_ptr<tensor_holder> original_w = make_intrusive<tensor_holder>(Tensor::Null());
    const SizeRef image_shape = image.shape().clone();
    const SizeRef kernel_shape = kernel.shape().clone();
    TensorGrad result(::nt::functional::conv_transpose3d(image.tensor, kernel.tensor, stride, padding, output_padding, dilation, groups, original_x, original_w));
    result.track_tensors(image, kernel);
    result.create_backward_function([image_shape, kernel_shape, stride, padding, output_padding, dilation, groups](const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& parents,
						intrusive_ptr<tensor_holder> img, intrusive_ptr<tensor_holder> kern){
        ::nt::functional::convt_dimage(grad, kern->tensor, parents[0]->grad->tensor, 
                                     {kernel_shape[-3], kernel_shape[-2], kernel_shape[-1]},
                                     {stride[0], stride[1], stride[2]},
                                     {padding[0], padding[1], padding[2]},
                                     {output_padding[0], output_padding[1], output_padding[2]},
                                     {dilation[0], dilation[1], dilation[2]},
                                     groups);
        ::nt::functional::convt_dkernel(grad, img->tensor, parents[1]->grad->tensor, {padding[0], padding[1], padding[2]}, {image_shape[-3], image_shape[-2], image_shape[-3]}, groups);
    }, original_x, original_w);
    return std::move(result);
}



} // namespace functional
} // namespace nt
