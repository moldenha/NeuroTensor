#include "min_max.h"
#include "../cpu/min_max.h"
#include "../functional.h"
#include "exceptions.hpp"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

Tensor clamp(const Tensor& x, std::optional<Scalar> min, std::optional<Scalar> max){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
	Tensor out = x.clone();
	if(min && max){
        cpu::_clamp(out.arr_void(), min.value(), max.value());
		return std::move(out);
	}
	else if(min)
		out[out < min.value()] = 0;
	else if(max)
		out[out > max.value()] = max.value();
	return std::move(out);
}
Tensor relu(const Tensor& x){return clamp(x, 0);}

Tensor silu(const Tensor& x){
	return x * sigmoid(x);
}

Tensor dsilu(const Tensor& x){
	Tensor sigmoid_x = sigmoid(x);
	Tensor grad = sigmoid_x * (1 + x * (1 - sigmoid_x));
	return std::move(grad);
}



Tensor gelu(const Tensor& x){
	Scalar sqrt_2_pi = std::sqrt(2.0 / M_PI);
	return 0.5 * x * (1.0 + tanh(sqrt_2_pi * (x + 0.044715 * std::pow(x, 3))));
}

Tensor dgelu(const Tensor& x) {
    const Scalar sqrt_2_pi(std::sqrt(2.0 / M_PI));
    const Scalar c(0.044715);

    Tensor z = sqrt_2_pi * (x + c * std::pow(x, 3));
    // Compute tanh(z) and its derivative
    z = tanh(z);
    Tensor tanh_derivative = 1 - (z * z);

    // Gradient of z with respect to x
    Tensor dz_dx = sqrt_2_pi * (1 + 3 * c.to<double>() * x * x);

    // Final gradient
    return 0.5 * (1 + z) + 0.5 * x * tanh_derivative * dz_dx;
}

inline bool is_complex(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isComplex();
}

inline bool is_floating(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isFloatingPoint();
}

inline bool is_integral(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isIntegral();
}

inline bool is_bool(const std::vector<Scalar>& scalars)noexcept {
    return scalars[0].isBoolean();
}


inline bool same_type(std::vector<Scalar>& scalars){
    if(is_complex(scalars)){
        for(const auto& s : scalars){
            if(!s.isComplex())
                return false;
        }
        return true;
    }
    if(is_floating(scalars)){
        for(const auto& s : scalars){
            if(!s.isFloatingPoint())
                return false;
        }
        return true;
    }
    if(is_integral(scalars)){
        for(const auto& s : scalars){
            if(!s.isIntegral())
                return false;
        }
        return true;
    }
    if(is_bool(scalars)){
        for(const auto& s : scalars){
            if(!s.isBoolean())
                return false;
        }
        return true;
    }
    return false;
}

template<typename Comp, typename T>
inline Scalar compare_scalars_type(const std::vector<Scalar>& scs, Comp&& func){
    T start = scs[0].to<T>();
    for(size_t i = 1; i < scs.size(); ++i){
        start = func(start, scs[i].to<T>());
    }
    return Scalar(start);
}

template<typename Comp>
inline Scalar compare_scalars_complex(const std::vector<Scalar>& scs, Comp&& func){
    return compare_scalars_type<Comp, complex_128>(scs, std::forward<Comp&&>(func)); 
    // Scalar start = scs[0];
    // complex_128 start = = scs[0].to<complex_128>();
    // for(size_t i = 1; i < scs.size(); ++i){
    //     start = func(scs[i].to<complex_128>(), start);
    // }
    // return Scalar(start);
}

template<typename Comp>
inline Scalar compare_scalars_integral(const std::vector<Scalar>& scs, Comp&& func){
    return compare_scalars_type<Comp, int64_t>(scs, std::forward<Comp&&>(func)); 
    // Scalar start = scs[0];
    // int64_t start = scs[0].to<int64_t>();
    // for(size_t i = 1; i < scs.size(); ++i){
    //     start = func(scs[i].v.to<int64_t>(), start.v.i);
    // }
}


template<typename Comp>
inline Scalar compare_scalars_floating(const std::vector<Scalar>& scs, Comp&& func){
    return compare_scalars_type<Comp, double>(scs, std::forward<Comp&&>(func)); 
    // Scalar start = scs[0];
    // for(size_t i = 1; i < scs.size(); ++i){
    //     start = func(scs[i].v.d, start.v.d);
    // }
}

Scalar min(std::vector<Scalar> scalars){
    if(scalars.size() == 1){return scalars[0];}
    utils::throw_exception(scalars.size() > 0, "Cannot find min of no scalars");
    utils::throw_exception(same_type(scalars), "Expected to compare all scalars of the same type");
    utils::throw_exception(!is_bool(scalars), "Cannot find min of bools");
    auto func = [](auto& a, auto b){return std::min(a, b);};
    if(is_complex(scalars)){
        return compare_scalars_complex(scalars, func); 
    }
    if(is_integral(scalars)){
        return compare_scalars_integral(scalars, func); 
    }
    return compare_scalars_floating(scalars, func); 
}
Scalar max(std::vector<Scalar> scalars){
    if(scalars.size() == 1){return scalars[0];}
    utils::throw_exception(scalars.size() > 0, "Cannot find max of no scalars");
    utils::throw_exception(same_type(scalars), "Expected to compare all scalars of the same type");
    utils::throw_exception(!is_bool(scalars), "Cannot find max of bools");
    auto func = [](auto& a, auto b){return std::max(a, b);};
    if(is_complex(scalars)){
        return compare_scalars_complex(scalars, func); 
    }
    if(is_integral(scalars)){
        return compare_scalars_integral(scalars, func); 
    }
    return compare_scalars_floating(scalars, func); 

}

inline bool same_type(const std::vector<Tensor>& tensors){
    DType dt = tensors[0].dtype;
    for(size_t i = 1; i < tensors.size(); ++i){
        if(tensors[i].dtype != dt){return false;}
    }
    return true;
}

inline bool same_shape(const std::vector<Tensor>& tensors){
    const SizeRef& shape = tensors[0].shape();
    for(size_t i = 1; i < tensors.size(); ++i){
        if(tensors[i].shape() != shape){return false;}
    }
    return true;
}


Tensor min(std::vector<Tensor> tensors){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find min of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for min");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for min");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for min got $", dt);
    Tensor out = tensors[0].clone();
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size()-1);
    for(size_t i = 1; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_min(out.arr_void(), arrvds);
    return std::move(out);
}

Tensor max(std::vector<Tensor> tensors){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find max of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for max");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for max");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for max got $", dt);
    Tensor out = tensors[0].clone();
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size()-1);
    for(size_t i = 1; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_max(out.arr_void(), arrvds);
    return std::move(out);
}


Tensor min(std::vector<Tensor> tensors, Scalar sc){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find min of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for min");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for min");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for min got $", dt);
    Tensor out(shape, dt);
    out = sc;
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size());
    for(size_t i = 0; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_min(out.arr_void(), arrvds);
    return std::move(out);

}
Tensor max(std::vector<Tensor> tensors, Scalar sc){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find max of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for max");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for max");
    const DType& dt = tensors[0].dtype;
    utils::throw_exception(dt != DType::Bool && dt != DType::TensorObj, 
                           "Expected to get dtype of number type for max got $", dt);
    Tensor out(shape, dt);
    out = sc;
    std::vector<ArrayVoid> arrvds;
    arrvds.reserve(tensors.size());
    for(size_t i = 0; i < tensors.size(); ++i){
        arrvds.emplace_back(tensors[i].arr_void().contiguous());
    }
    cpu::_max(out.arr_void(), arrvds);
    return std::move(out);
 
}

result_types::max<Tensor, Tensor> max_(const Tensor &_x) {
    if (_x.dtype == DType::TensorObj) {
        result_types::max<Tensor, Tensor> output(
            Tensor(_x.shape(), DType::TensorObj),
            Tensor(_x.shape(), DType::TensorObj));
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&output](auto begin, auto end) {
                    Tensor *v_begin =
                        reinterpret_cast<Tensor *>(output.values.data_ptr());
                    Tensor *i_begin =
                        reinterpret_cast<Tensor *>(output.indices.data_ptr());
                    for (; begin != end; ++begin, ++v_begin, ++i_begin) {
                        result_types::max<Tensor, Tensor> o = begin->max();
                        *v_begin = o.values;
                        *i_begin = o.indices;
                    }
                });
        return std::move(output);
    }
    Tensor outp(1, _x.dtype);
    Tensor indices(_x.shape(), DType::Bool);
    indices.fill_(false);
    outp = _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&indices](auto begin, auto end) -> Scalar {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            uint_bool_t *i_begin =
                reinterpret_cast<uint_bool_t *>(indices.data_ptr());
            value_t max_element = *begin;
            uint_bool_t *max_indice = i_begin;
            ++begin;
            for (; begin != end; ++begin, ++i_begin) {
                if (*begin > max_element) {
                    max_element = *begin;
                    max_indice = i_begin;
                }
            }
            *max_indice = uint_bool_t(true);
            return max_element;
        });
    return result_types::max<Tensor, Tensor>(std::move(outp),
                                             std::move(indices));
}

result_types::max<Tensor, Tensor> min_(const Tensor &_x) {
    if (_x.dtype == DType::TensorObj) {
        result_types::max<Tensor, Tensor> output(
            Tensor(_x.shape(), DType::TensorObj),
            Tensor(_x.shape(), DType::TensorObj));
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&output](auto begin, auto end) {
                    Tensor *v_begin =
                        reinterpret_cast<Tensor *>(output.values.data_ptr());
                    Tensor *i_begin =
                        reinterpret_cast<Tensor *>(output.indices.data_ptr());
                    for (; begin != end; ++begin, ++v_begin, ++i_begin) {
                        result_types::max<Tensor, Tensor> o = begin->min();
                        *v_begin = o.values;
                        *i_begin = o.indices;
                    }
                });
        return std::move(output);
    }
    Tensor outp(1, _x.dtype);
    Tensor indices(_x.shape(), DType::Bool);
    indices.fill_(false);
    outp = _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>()(
        [&indices](auto begin, auto end) -> Scalar {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            uint_bool_t *i_begin =
                reinterpret_cast<uint_bool_t *>(indices.data_ptr());
            value_t min_element = *begin;
            uint_bool_t *min_indice = i_begin;
            ++begin;
            for (; begin != end; ++begin, ++i_begin) {
                if (*begin < min_element) {
                    min_element = *begin;
                    min_indice = i_begin;
                }
            }
            *min_indice = uint_bool_t(true);
            return min_element;
        });
    return result_types::max<Tensor, Tensor>(std::move(outp),
                                             std::move(indices));
}

//result_types::max<Tensor, Tensor> max_(const Tensor &_x,
//                                       Tensor::size_value_t dim) {
//    dim = dim < 0 ? dim + _x.dims() : dim;
//    if (_x.dtype == DType::TensorObj) {
//        result_types::max<Tensor, Tensor> output(
//            Tensor(_x.shape(), DType::TensorObj),
//            Tensor(_x.shape(), DType::TensorObj));
//        _x.arr_void()
//            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
//                [&output, &dim](auto begin, auto end) {
//                    Tensor *v_begin =
//                        reinterpret_cast<Tensor *>(output.values.data_ptr());
//                    Tensor *i_begin =
//                        reinterpret_cast<Tensor *>(output.indices.data_ptr());
//                    for (; begin != end; ++begin, ++v_begin, ++i_begin) {
//                        result_types::max<Tensor, Tensor> o = begin->max(dim);
//                        *v_begin = o.values;
//                        *i_begin = o.indices;
//                    }
//                });
//        return output;
//    }
//    //consider making these part of the cpu_ functions
//    //it wouldn't be too hard to implement
//    Tensor bools(_x.shape(), DType::Bool);
//    SizeRef o_shape = _x.shape().delete_index(dim);
//    if (dim == _x.dims() - 1) {
//        _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
//            [&bools](auto begin, auto end) {
//                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
//                uint_bool_t *b_begin =
//                    reinterpret_cast<uint_bool_t *>(bools.data_ptr());
//                const Tensor::size_value_t &rows = bools.shape()[-1];
//                while (begin != end) {
//                    uint_bool_t *b_max_ele = b_begin;
//                    value_t max_ele = *begin;
//                    auto current_end = begin + rows;
//                    ++begin;
//                    ++b_begin;
//                    for (; begin != current_end; ++begin, ++b_begin) {
//                        if (*begin > max_ele) {
//                            max_ele = *begin;
//                            b_max_ele = b_begin;
//                        }
//                    }
//                    *b_max_ele = uint_bool_t(true);
//                }
//            });
//        return result_types::max<Tensor, Tensor>((_x)[bools].view(o_shape),
//                                                 std::move(bools));
//    }
//    Tensor bools_t = bools.transpose(dim, -1);
//    Tensor _t = _x.transpose(dim, -1);
//    _t.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
//        [&bools_t](auto begin, auto end) {
//            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
//            auto b_begin =
//                bools_t.arr_void().get_bucket().begin<3, uint_bool_t>();
//            const Tensor::size_value_t &rows = bools_t.shape()[-1];
//            while (begin != end) {
//                auto b_max_ele = b_begin;
//                value_t max_ele = *begin;
//                auto current_end = begin + rows;
//                ++begin;
//                ++b_begin;
//                for (; begin != current_end; ++begin, ++b_begin) {
//                    if (*begin > max_ele) {
//                        max_ele = *begin;
//                        b_max_ele = b_begin;
//                    }
//                }
//                *b_max_ele = uint_bool_t(true);
//            }
//        });

//    return result_types::max<Tensor, Tensor>((_x)[bools].view(o_shape),
//                                             std::move(bools));
//    /* if(dim == dims()-1){ */

//    /* } */

//    /* size_value_t total_size = shape().flatten(0,dim)[0]; */
//    /* Tensor outp(shape()[my_range(0, dim)], dtype); */
//    /* const Tensor split = this->split_axis(dim); */
//    /* outp._vals.execute_function<WRAP_DTYPES<RealNumberTypesL>>()([](auto
//     * begin, auto end, const Tensor* vals){ */
//    /* 			using value_t =
//     * utils::IteratorBaseType_t<decltype(begin)>; */
//    /* 			for(;begin != end; ++begin, ++vals){ */
//    /* 				*begin = vals->max().toScalar().to<value_t>();
//     */
//    /* 			} */
//    /* 		}, reinterpret_cast<const Tensor*>(split.data_ptr())); */
//    /* return std::move(outp); */
//}

Tensor& max_(const Tensor &_x, Tensor& bools) {
    if (_x.dtype == DType::TensorObj) {
        utils::throw_exception(bools.dtype == DType::TensorObj,
                               "Expected indices to be tensor obj for a tensor obj comparison");
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&bools](auto begin, auto end) {
                    bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                    [&](auto i_begin, auto i_end){
                    for (; begin != end; ++begin, ++i_begin) {
                        Tensor& indice = *i_begin;
                        const Tensor& val = *begin;
                        max_indices(val, indice);
                    }});
                });
        return bools;
    }
    _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&bools](auto begin, auto end) {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
            [&](auto i_begin, auto i_end){
            value_t max_element = *begin;
            auto max_indice = i_begin;
            ++begin;
            for (; begin != end; ++begin, ++i_begin) {
                if (*begin > max_element) {
                    max_element = *begin;
                    max_indice = i_begin;
                }
            }
            *max_indice = uint_bool_t(true);
        });
        });
    return bools;
}



Tensor& min_(const Tensor &_x, Tensor& bools) {
    if (_x.dtype == DType::TensorObj) {
        utils::throw_exception(bools.dtype == DType::TensorObj,
                               "Expected indices to be tensor obj for a tensor obj comparison");
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&bools](auto begin, auto end) {
                    bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                    [&](auto i_begin, auto i_end){
                    for (; begin != end; ++begin, ++i_begin) {
                        min_indices(*begin, *i_begin);
                    }});
                });
        return bools;
    }
    _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        [&bools](auto begin, auto end) {
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
            [&](auto i_begin, auto i_end){
            value_t min_element = *begin;
            auto min_indice = i_begin;
            ++begin;
            for (; begin != end; ++begin, ++i_begin) {
                if (*begin < min_element) {
                    min_element = *begin;
                    min_indice = i_begin;
                }
            }
            *min_indice = uint_bool_t(true);
        });
        });
    return bools;
}

Tensor& max_(const Tensor &_x,
                                       Tensor::size_value_t dim,
                                       Tensor& bools) {
    dim = dim < 0 ? dim + _x.dims() : dim;
    if (_x.dtype == DType::TensorObj) {
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&bools, &dim](auto begin, auto end) {
                    bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                    [&](auto i_begin, auto i_end){
                    for (; begin != end; ++begin, ++i_begin) {
                        max_indices(*begin, *i_begin, dim);
                    }});
                });
        return bools;
    }
    //consider making these part of the cpu_ functions
    //it wouldn't be too hard to implement
    // Tensor bools(_x.shape(), DType::Bool);
    //bools.fill_(false);
    // SizeRef o_shape = _x.shape().delete_index(dim);
    if (dim == _x.dims() - 1) {
        _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [&bools](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
                    [&](auto b_begin, auto b_end){
                    const Tensor::size_value_t &rows = bools.shape()[-1];
                    while (begin != end) {
                        auto b_max_ele = b_begin;
                        value_t max_ele = *begin;
                        auto current_end = begin + rows;
                        ++begin;
                        ++b_begin;
                        for (; begin != current_end; ++begin, ++b_begin) {
                            if (*begin > max_ele) {
                                max_ele = *begin;
                                b_max_ele = b_begin;
                            }
                        }
                        *b_max_ele = uint_bool_t(true);
                    }
                    });
                });
        return bools;
    }
    Tensor bools_t = bools.transpose(dim, -1);
    Tensor _t = _x.transpose(dim, -1);
    max_(_t, _x.dims()-1, bools_t);
    return bools;
    // _t.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
        // [&bools_t](auto begin, auto end) {
        //     using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        //     // auto b_begin =
        //     //     bools_t.arr_void().get_bucket().begin<3, uint_bool_t>();
        //     bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
        //         [&](auto b_begin, auto b_end){
        //     const Tensor::size_value_t &rows = bools_t.shape()[-1];
        //     while (begin != end) {
        //         auto b_max_ele = b_begin;
        //         value_t max_ele = *begin;
        //         auto current_end = begin + rows;
        //         ++begin;
        //         ++b_begin;
        //         for (; begin != current_end; ++begin, ++b_begin) {
        //             if (*begin > max_ele) {
        //                 max_ele = *begin;
        //                 b_max_ele = b_begin;
        //             }
        //         }
        //         *b_max_ele = uint_bool_t(true);
        //     }});
        // });

    // return bools;
}



Tensor& min_(const Tensor &_x,
                                       Tensor::size_value_t dim,
                                       Tensor& bools) {
    dim = dim < 0 ? dim + _x.dims() : dim;
    if (_x.dtype == DType::TensorObj) {
        _x.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                [&bools, &dim](auto begin, auto end) {
                    bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj>>>(
                    [&](auto i_begin, auto i_end){
                    for (; begin != end; ++begin, ++i_begin) {
                        min_indices(*begin, *i_begin, dim);
                    }});
                });
        return bools;
    }
    //consider making these part of the cpu_ functions
    //it wouldn't be too hard to implement
    // Tensor bools(_x.shape(), DType::Bool);
    //bools.fill_(false);
    // SizeRef o_shape = _x.shape().delete_index(dim);
    if (dim == _x.dims() - 1) {
        _x.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
            [&bools](auto begin, auto end) {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
                    [&](auto b_begin, auto b_end){
                    const Tensor::size_value_t &rows = bools.shape()[-1];
                    while (begin != end) {
                        auto b_min_ele = b_begin;
                        value_t min_ele = *begin;
                        auto current_end = begin + rows;
                        ++begin;
                        ++b_begin;
                        for (; begin != current_end; ++begin, ++b_begin) {
                            if (*begin < min_ele) {
                                min_ele = *begin;
                                b_min_ele = b_begin;
                            }
                        }
                        *b_min_ele = uint_bool_t(true);
                    }
                    });
                });
        return bools;
    }
    Tensor bools_t = bools.transpose(dim, -1);
    Tensor _t = _x.transpose(dim, -1);
    min_(_t, _x.dims()-1, bools_t);
    return bools;
    // _t.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL>>(
    //     [&bools_t](auto begin, auto end) {
    //         using value_t = utils::IteratorBaseType_t<decltype(begin)>;
    //         // auto b_begin =
    //         //     bools_t.arr_void().get_bucket().begin<3, uint_bool_t>();
    //         bools.arr_void().execute_function<WRAP_DTYPES<DTypeEnum<DType::Bool>>>(
    //             [&](auto b_begin, auto b_end){
    //         const Tensor::size_value_t &rows = bools_t.shape()[-1];
    //         while (begin != end) {
    //             auto b_max_ele = b_begin;
    //             value_t max_ele = *begin;
    //             auto current_end = begin + rows;
    //             ++begin;
    //             ++b_begin;
    //             for (; begin != current_end; ++begin, ++b_begin) {
    //                 if (*begin > max_ele) {
    //                     max_ele = *begin;
    //                     b_max_ele = b_begin;
    //                 }
    //             }
    //             *b_max_ele = uint_bool_t(true);
    //         }});
    //     });

    // return bools;
}


Tensor& max_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(tensor, indices);
    utils::throw_exception(indices.is_mutable(),
                           "Output indices from the max indices function must be mutable");
    utils::THROW_EXCEPTION(tensor.shape() == indices.shape(),
                           "The indices shape ($) must equal tensor shape ($)", indices.shape(), tensor.shape());
    utils::THROW_EXCEPTION((tensor.dtype == DType::TensorObj && indices.dtype == DType::TensorObj) || indices.dtype == DType::Bool,
                           "Expected max_indices to be dtype Bool or tensors of bools if the values are tensors but got $", indices.dtype);
    if(indices.dtype == DType::Bool){indices.fill_(false);}
    if(!list){
        return max_(tensor, indices);
    }
    max_(tensor, list[0], indices);
    //SizeRef o_shape = tensor.shape().delete_index(list[0]);
    for(auto begin = list->cbegin() + 1; begin != list->cend(); ++begin){
        max_(tensor, *begin, indices);
    }
    return indices;
}
Tensor& min_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(tensor, indices);
    utils::throw_exception(indices.is_mutable(),
                           "Output indices from the min indices function must be mutable");
    utils::throw_exception(tensor.shape() == indices.shape(),
                           "The indices shape ($) must equal tensor shape ($)", indices.shape(), tensor.shape());
    utils::throw_exception((tensor.dtype == DType::TensorObj && indices.dtype == DType::TensorObj) || indices.dtype == DType::Bool,
                           "Expected max_indices to be dtype Bool or tensors of bools if the values are tensors but got $", indices.dtype);
    if(indices.dtype == DType::Bool){indices.fill_(false);}
    if(!list){
        return min_(tensor, indices);
    }
    min_(tensor, list[0], indices);
    //SizeRef o_shape = tensor.shape().delete_index(list[0]);
    for(auto begin = list->cbegin() + 1; begin != list->cend(); ++begin){
        min_(tensor, *begin, indices);
    }
    return indices;
 
}

Tensor max_indices(const Tensor& tensor, utils::optional_list list){
    Tensor indices(tensor.shape(), DType::Bool);
    indices.fill_(false);
    max_indices(tensor, indices, list);
    return std::move(indices);
}

Tensor min_indices(const Tensor& tensor, utils::optional_list list){
    Tensor indices(tensor.shape(), DType::Bool);
    indices.fill_(false);
    min_indices(tensor, indices, list);
    return std::move(indices);
}


result_types::max<Tensor, Tensor> max(const Tensor& self, utils::optional_list list){
    if(!list){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(self);
        return max_(self);
    }
    Tensor indices = max_indices(self, list);
    SizeRef o_shape = self.shape().delete_index(list[0]);
    for(auto begin = list->cbegin() + 1; begin != list->cend(); ++begin){
        o_shape = o_shape.delete_index(*begin);
    }
    return result_types::max<Tensor, Tensor>(self[indices].view(o_shape).contiguous(), std::move(indices));
}

result_types::max<Tensor, Tensor> min(const Tensor& self, utils::optional_list list){
    if(!list){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(self);
        return min_(self);
    }
    Tensor indices = min_indices(self, list);
    SizeRef o_shape = self.shape().delete_index(list[0]);
    for(auto begin = list->cbegin() + 1; begin != list->cend(); ++begin){
        o_shape = o_shape.delete_index(*begin);
    }
    return result_types::max<Tensor, Tensor>(self[indices].view(o_shape).contiguous(), std::move(indices));
}

Tensor argmin(Tensor x){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    auto results = x.view(-1).min();
    Tensor out = where(results.indices.view(-1));
    if(out.dtype == DType::TensorObj){
        return out.item<Tensor>();
    }
    return std::move(out);
}
Tensor argmax(Tensor x){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    auto results = x.view(-1).max();
    Tensor out = where(results.indices.view(-1));
    if(out.dtype == DType::TensorObj){
        return out.item<Tensor>();
    }
    return std::move(out);
}
Tensor argmin(Tensor x, int64_t dim, bool keepdims){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    dim = dim < 0 ? dim + x.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < x.dims(), "Dim $ out of range for argmin of tensor $", (dim < 0 ? dim - x.dims() : dim), x.shape());
    auto results = x.min(dim);
    Tensor out = where(results.indices)[dim].item<Tensor>();
    if(!keepdims) return out;
    auto out_shape = x.shape().redo_index(dim, 1);
    // std::vector<SizeRef::ArrayRefInt::value_type> vec = x.shape().Vec();
    // for(size_t i = 0; i < vec.size(); ++i)
    //     if(i != dim)
    //         vec[i] = 1;
    return out.view(out_shape);
    
}
Tensor argmax(Tensor x, int64_t dim, bool keepdims){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    dim = dim < 0 ? dim + x.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < x.dims(), "Dim $ out of range for argmax of tensor $", (dim < 0 ? dim - x.dims() : dim), x.shape());
    auto results = x.max(dim);
    Tensor out = where(results.indices)[dim].item<Tensor>();
    if(!keepdims) return out;
    auto out_shape = x.shape().redo_index(dim, 1);
    // std::vector<SizeRef::ArrayRefInt::value_type> vec = x.shape().Vec();
    // vec[dim] = 1;
    // for(size_t i = 0; i < vec.size(); ++i)
    //     if(i != dim)
    //         vec[i] = 1;
    return out.view(out_shape);
  
}


}
}
