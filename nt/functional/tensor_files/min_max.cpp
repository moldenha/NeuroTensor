#include "min_max.h"
#include "../cpu/min_max.h"
#include "../functional.h"
#include "exceptions.hpp"
#include "../../dtype/ArrayVoid.hpp"
#include <algorithm>
#include <functional>
#include <set>

namespace nt{
namespace functional{

Tensor clamp(const Tensor& x, std::optional<Scalar> min, std::optional<Scalar> max){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
	Tensor out = x.clone();
	if(min && max){
        cpu::_clamp(out.arr_void(), min.value(), max.value());
		return std::move(out);
	}
	else if(min){
		cpu::_clamp_below(out.arr_void(), min.value());
    }
	else if(max){
		cpu::_clamp_above(out.arr_void(), max.value());
    }
	return std::move(out);
}

Tensor& clamp_(Tensor& x, std::optional<Scalar> min, std::optional<Scalar> max){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    utils::THROW_EXCEPTION(x.is_mutable(), "Cannot perform operation clamp_ on an immutable tensor");

	// Tensor out = x.clone();
	if(min && max){
        cpu::_clamp(x.arr_void(), min.value(), max.value());
		return x;
	}
	else if(min){
		cpu::_clamp_below(x.arr_void(), min.value());
    }
	else if(max){
		cpu::_clamp_above(x.arr_void(), max.value());
    }
	return x;
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

Scalar minimum(std::vector<Scalar> scalars){
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
Scalar maximum(std::vector<Scalar> scalars){
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
    DType dt = tensors[0].dtype();
    for(size_t i = 1; i < tensors.size(); ++i){
        if(tensors[i].dtype() != dt){return false;}
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


Tensor minimum(std::vector<Tensor> tensors){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find min of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for min");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for min");
    const DType& dt = tensors[0].dtype();
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

Tensor maximum(std::vector<Tensor> tensors){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find max of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for max");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for max");
    const DType& dt = tensors[0].dtype();
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


Tensor minimum(std::vector<Tensor> tensors, Scalar sc){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find min of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for min");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for min");
    const DType& dt = tensors[0].dtype();
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
Tensor maximum(std::vector<Tensor> tensors, Scalar sc){
    for(const auto& x : tensors)
        _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    if(tensors.size() == 1){return tensors[0];}
    utils::throw_exception(tensors.size() > 0, "Cannot find max of no tensors");
    const SizeRef& shape = tensors[0].shape();
    utils::throw_exception(same_type(tensors), "Expected to compare all tensors of the same dtype for max");
    utils::throw_exception(same_shape(tensors), "Expected to compare all tensors of the same shape for max");
    const DType& dt = tensors[0].dtype();
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
    if (_x.dtype() == DType::TensorObj) {
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
    Tensor outp(1, _x.dtype());
    Tensor indices(_x.shape(), DType::Bool);
    indices.fill_(false);
    outp = cpu::_max_scalar(_x.arr_void(), indices.arr_void());
    return result_types::max<Tensor, Tensor>(std::move(outp),
                                             std::move(indices));
}

result_types::max<Tensor, Tensor> min_(const Tensor &_x) {
    if (_x.dtype() == DType::TensorObj) {
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
                        result_types::max<Tensor, Tensor> o = min_(*begin);
                        *v_begin = o.values;
                        *i_begin = o.indices;
                    }
                });
        return std::move(output);
    }
    Tensor outp(1, _x.dtype());
    Tensor indices(_x.shape(), DType::Bool);
    indices.fill_(false);
    outp = cpu::_min_scalar(_x.arr_void(), indices.arr_void());
    return result_types::max<Tensor, Tensor>(std::move(outp),
                                             std::move(indices));
}

//result_types::max<Tensor, Tensor> max_(const Tensor &_x,
//                                       Tensor::size_value_t dim) {
//    dim = dim < 0 ? dim + _x.dims() : dim;
//    if (_x.dtype() == DType::TensorObj) {
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
//    /* Tensor outp(shape()[range_(0, dim)], dtype); */
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

    if (_x.dtype() == DType::TensorObj) {
        utils::throw_exception(bools.dtype() == DType::TensorObj,
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
    // only called internally
    // utils::throw_exception(_x.numel() == bools.numel(), "Cannot get bools of max from tensor with $ elements to bools tensor with $ elements", _x.numel(), bools.numel());
    Scalar max = cpu::_max_scalar(_x.arr_void(), bools.arr_void());
    return bools;
}



Tensor& min_(const Tensor &_x, Tensor& bools) {
    if (_x.dtype() == DType::TensorObj) {
        utils::throw_exception(bools.dtype() == DType::TensorObj,
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
    Scalar max = cpu::_min_scalar(_x.arr_void(), bools.arr_void());
    return bools;
}

Tensor& max_(const Tensor &_x,
                                       Tensor::size_value_t dim,
                                       Tensor& bools) {
    dim = dim < 0 ? dim + _x.dims() : dim;
    if (_x.dtype() == DType::TensorObj) {
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
        cpu::_max_strided(_x.arr_void(), bools.arr_void(), bools.shape()[-1]);
        return bools;
    }
    Tensor bools_t = bools.transpose(dim, -1);
    Tensor _t = _x.transpose(dim, -1);
    max_(_t, _x.dims()-1, bools_t);
    return bools;
}



Tensor& min_(const Tensor &_x,
                                       Tensor::size_value_t dim,
                                       Tensor& bools) {
    dim = dim < 0 ? dim + _x.dims() : dim;
    if (_x.dtype() == DType::TensorObj) {
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
        cpu::_min_strided(_x.arr_void(), bools.arr_void(), bools.shape()[-1]);
        return bools;
    }
    Tensor bools_t = bools.transpose(dim, -1);
    Tensor _t = _x.transpose(dim, -1);
    min_(_t, _x.dims()-1, bools_t);
    return bools;
}


std::vector<Tensor::size_value_t> get_max_correct_perms(const Tensor& tensor, utils::optional_list list){
    //first going to check if the list has all the elements right after one another and if they start at the back
    //return an empty vector so that the function knows it doesn't have to call a permute
    const int64_t& dims = tensor.dims();
    //fix the list
    std::for_each(list.begin(), list.end(), [&dims](auto& val){val = (val < 0) ? dims + val : val;});
    //sort from largest to smallest
    std::sort(list.begin(), list.end(), std::greater<int64_t>());
    if(list[0] == (dims-1)){
        //this is the check
        auto begin_f = list->begin();
        auto begin = begin_f+1;
        auto end = list->end();
        bool dont_permute = true;
        //if it is true, begin is one before begin_f so it should be one larger (after sorting)
        for(;begin != end; ++begin, ++begin_f){
            if((*begin - 1) != *begin_f){
                dont_permute = false;
            }
        }
        if(dont_permute){return std::vector<Tensor::size_value_t>();}
    }
    std::vector<Tensor::size_value_t> perms;
    perms.reserve(tensor.dims());
    for(int64_t i = 0; i < tensor.dims(); ++i){
        bool inside = false;
        for(auto begin = list->cbegin(); begin != list->cend(); ++begin){
            if(*begin == i){inside = true; break;}
        }
        if(!inside)
            perms.push_back(i);
    }
    for(auto begin = list->cbegin(); begin != list->cend(); ++begin){
        perms.push_back(*begin);
    }
    return std::move(perms);
}

Tensor& max_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(tensor, indices);
    utils::throw_exception(indices.is_mutable(),
                           "Output indices from the max indices function must be mutable");
    utils::THROW_EXCEPTION(tensor.shape() == indices.shape(),
                           "The indices shape ($) must equal tensor shape ($)", indices.shape(), tensor.shape());
    utils::THROW_EXCEPTION((tensor.dtype() == DType::TensorObj && indices.dtype() == DType::TensorObj) || indices.dtype() == DType::Bool,
                           "Expected max_indices to be dtype Bool or tensors of bools if the values are tensors but got $", indices.dtype());
    if(indices.dtype() == DType::Bool){indices.fill_(false);}
    if(!list || list->size() == tensor.dims()){
        return max_(tensor, indices);
    }
    if(list->size() == 1){
        max_(tensor, list[0], indices);
        return indices;
    }
    std::vector<Tensor::size_value_t> perms = get_max_correct_perms(tensor, list);
    Tensor _input = perms.empty() ? tensor.flatten((-1) * (list->size()), -1) : permute(tensor, perms).flatten((-1) * (list->size()), -1);
    Tensor _indices = perms.empty() ? indices.flatten((-1) * (list->size()), -1) : permute(indices, perms).flatten((-1) * (list->size()), -1);
    max_(_input, -1, _indices);
    return indices;
}
Tensor& min_indices(const Tensor& tensor, Tensor& indices, utils::optional_list list){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(tensor, indices);
    utils::throw_exception(indices.is_mutable(),
                           "Output indices from the min indices function must be mutable");
    utils::throw_exception(tensor.shape() == indices.shape(),
                           "The indices shape ($) must equal tensor shape ($)", indices.shape(), tensor.shape());
    utils::throw_exception((tensor.dtype() == DType::TensorObj && indices.dtype() == DType::TensorObj) || indices.dtype() == DType::Bool,
                           "Expected max_indices to be dtype Bool or tensors of bools if the values are tensors but got $", indices.dtype());
    if(indices.dtype() == DType::Bool){indices.fill_(false);}
    if(!list || list->size() == tensor.dims()){
        return min_(tensor, indices);
    }
    if(list->size() == 1){
        min_(tensor, list[0], indices);
        return indices;
    }

    std::vector<Tensor::size_value_t> perms = get_max_correct_perms(tensor, list);
    Tensor _input = perms.empty() ? tensor.flatten((-1) * (list->size()), -1) : permute(tensor, perms).flatten((-1) * (list->size()), -1);
    Tensor _indices = perms.empty() ? indices.flatten((-1) * (list->size()), -1) : permute(indices, perms).flatten((-1) * (list->size()), -1);

    min_(_input, -1, _indices);
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

//the list is the dimensions to get the max along
result_types::max<Tensor, Tensor> max(const Tensor& self, utils::optional_list list, bool keepdim){
    if(!list){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(self);
        return max_(self);
    }
    const int64_t& dims = self.dims();
    std::for_each(list.begin(), list.end(), [&dims](auto& val){val = (val < 0) ? val + dims : val;});
    Tensor indices = max_indices(self, list);

    if(keepdim){
        SizeRef o_shape = self.shape().redo_index(list[0], 1);
        for(auto begin = list->cbegin() + 1; begin != list->cend(); ++begin){
            o_shape = o_shape.redo_index(*begin, 1);
        }
        return result_types::max<Tensor, Tensor>(self[indices].view(o_shape).contiguous(), std::move(indices));
    }
    else{
        std::set<int64_t> remove_set(list.cbegin(), list.cend());
        std::vector<int64_t> n_shape;
        n_shape.reserve(self.dims() - list->size());
        const auto& shape = self.shape();
        for(int64_t i = 0; i < shape.size(); ++i){
            if(remove_set.find(i) == remove_set.end()) {
                n_shape.push_back(shape[i]);
            }
        }
        return result_types::max<Tensor, Tensor>(self[indices].view(SizeRef(std::move(n_shape))).contiguous(), std::move(indices));
    }
}

result_types::max<Tensor, Tensor> min(const Tensor& self, utils::optional_list list, bool keepdim){
    if(!list){
        _NT_FUNCTIONAL_ALWAYS_CHECK_(self);
        return min_(self);
    }
    const int64_t& dims = self.dims();
    std::for_each(list.begin(), list.end(), [&dims](auto& val){val = (val < 0) ? val + dims : val;});
    Tensor indices = min_indices(self, list);
    if(keepdim){
        SizeRef o_shape = self.shape().redo_index(list[0], 1);
        for(auto begin = list->cbegin() + 1; begin != list->cend(); ++begin){
            o_shape = o_shape.redo_index(*begin, 1);
        } 
        return result_types::max<Tensor, Tensor>(self[indices].view(o_shape).contiguous(), std::move(indices));
    }else{
        std::set<int64_t> remove_set(list.cbegin(), list.cend());
        std::vector<int64_t> n_shape;
        n_shape.reserve(self.dims() - list->size());
        const auto& shape = self.shape();
        for(int64_t i = 0; i < shape.size(); ++i){
            if(remove_set.find(i) == remove_set.end()) {
                n_shape.push_back(shape[i]);
            }
        }
        return result_types::max<Tensor, Tensor>(self[indices].view(SizeRef(std::move(n_shape))).contiguous(), std::move(indices));
    }
}

Tensor argmin(Tensor x){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    // auto results = x.view(-1).min();
    auto results = min_(x);
    Tensor out = where(results.indices.view(-1));
    if(out.dtype() == DType::TensorObj){
        return out.item<Tensor>();
    }
    return std::move(out);
}
Tensor argmax(Tensor x){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    auto results = x.view(-1).max();
    Tensor out = where(results.indices.view(-1));
    if(out.dtype() == DType::TensorObj){
        return out.item<Tensor>();
    }
    return std::move(out);
}
Tensor argmin(Tensor x, int64_t dim, bool keepdims){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(x);
    dim = dim < 0 ? dim + x.dims() : dim;
    utils::throw_exception(dim >= 0 && dim < x.dims(), "Dim $ out of range for argmin of tensor $", (dim < 0 ? dim - x.dims() : dim), x.shape());
    result_types::max<Tensor, Tensor> results = x.min(dim);
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
