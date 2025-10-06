#ifndef NT_FUNCTIONAL_TENSOR_FILES_DILATE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_DILATE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"
#include "../../utils/collect_ri.hpp"
#include <vector>

namespace nt {
namespace functional {

// NEUROTENSOR_API Tensor undilate_(const Tensor&, Tensor::size_value_t);
// NEUROTENSOR_API Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
// NEUROTENSOR_API Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor undilate_(const Tensor&, std::vector<Tensor::size_value_t>, bool test = false);
template<typename... Args>
inline Tensor undilate_(const Tensor& t, Tensor::size_value_t i, Args&&... args){
    std::vector<Tensor::size_value_t> vec;
    vec.reserve(sizeof...(Args) + 1);
    utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
    return undilate_(t, std::move(vec), false);
}

// NEUROTENSOR_API Tensor undilate(const Tensor&, Tensor::size_value_t);
// NEUROTENSOR_API Tensor undilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
// NEUROTENSOR_API Tensor undilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor undilate(const Tensor&, std::vector<Tensor::size_value_t>);
template<typename... Args>
inline Tensor undilate(const Tensor& t, Tensor::size_value_t i, Args&&... args){
    std::vector<Tensor::size_value_t> vec;
    vec.reserve(sizeof...(Args) + 1);
    utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
    return undilate(t, std::move(vec));
}



// NEUROTENSOR_API Tensor dilate(const Tensor&, Tensor::size_value_t);
// NEUROTENSOR_API Tensor dilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
// NEUROTENSOR_API Tensor dilate(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor dilate(const Tensor&, std::vector<Tensor::size_value_t>, bool test = false);
template<typename... Args>
inline Tensor dilate(const Tensor& t, Tensor::size_value_t i, Args&&... args){
    std::vector<Tensor::size_value_t> vec;
    vec.reserve(sizeof...(Args) + 1);
    utils::collect_integers_impl(vec, i, std::forward<Args>(args)...);
    return dilate(t, std::move(vec), false);
}

}
}

#endif
