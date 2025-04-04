#ifndef __NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_HPP__
#define __NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_HPP__

#include "../../utils/type_traits.h"
#include "min_max.h"
#include <type_traits>

namespace nt {
namespace functional {
namespace min_max_detail {

inline void to_vec_tensor_sub(std::vector<Tensor> &out_tensors) { ; }

template <typename T, typename... Args>
inline void to_vec_tensor_sub(std::vector<Tensor> &out_tensors, T &&arg,
                              Args &&...args) {
    static_assert(std::is_same_v<std::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<std::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < std::remove_cvref_t < T >>,
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, Tensor>) {
        out_tensors.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_tensor_sub(out_tensors, std::forward<Args &&>(args)...);
}

template <typename T, typename... Args>
inline std::vector<Tensor> to_vec_tensor(T &&arg, Args &&...args) {
    static_assert(std::is_same_v<std::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<std::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < std::remove_cvref_t < T >>, 
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    std::vector<Tensor> out_tensors;
    out_tensors.reserve(sizeof...(Args) + 1);
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, Tensor>) {
        out_tensors.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_tensor_sub(out_tensors, std::forward<Args &&>(args)...);
    return std::move(out_tensors);
}

inline void to_vec_scalar_sub(std::vector<Scalar> &out_scalarss) { ; }

template <typename T, typename... Args>
inline void to_vec_scalar_sub(std::vector<Scalar> &out_scalars, T &&arg,
                              Args &&...args) {
    static_assert(std::is_same_v<std::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<std::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < std::remove_cvref_t < T >>, 
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, Scalar>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    } else if constexpr (utils::is_scalar_value_v<std::remove_cvref_t<T>>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_scalar_sub(out_scalars, std::forward<Args &&>(args)...);
}

template <typename T, typename... Args>
inline std::vector<Scalar> to_vec_scalar(T &&arg, Args &&...args) {
    static_assert(std::is_same_v<std::remove_cvref_t<T>, Tensor> ||
                  std::is_same_v<std::remove_cvref_t<T>, Scalar> ||
                  utils::is_scalar_value_v < std::remove_cvref_t < T >>, 
                      "Expected all types to be a tensor or scalar when "
                      "getting min or max using functional");
    std::vector<Scalar> out_scalars;
    out_scalars.reserve(sizeof...(Args) + 1);
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, Scalar>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    } else if constexpr (utils::is_scalar_value_v<std::remove_cvref_t<T>>) {
        out_scalars.emplace_back(std::forward<T &&>(arg));
    }
    to_vec_scalar_sub(out_scalars, std::forward<Args &&>(args)...);
    return std::move(out_scalars);
}
} // namespace min_max_detail

// can take arbitraty scalars and tensors and find the max
template <typename... Args> inline Tensor max(Args &&...args) {
    std::vector<Tensor> tensors =
        min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
    std::vector<Scalar> scalars =
        min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);

    if (scalars.size() == 0) {
        return max(std::move(tensors));
    } else if (tensors.size() == 0) {
        Tensor out({1}, scalars[0].type());
        out = max(std::move(scalars));
        return std::move(out);
    }
    Scalar max_s = max(std::move(scalars));
    return max(std::move(tensors), max_s);
}

// can take arbitraty scalars and tensors and find the min
template <typename... Args> inline Tensor min(Args &&...args) {
    std::vector<Tensor> tensors =
        min_max_detail::to_vec_tensor(std::forward<Args &&>(args)...);
    std::vector<Scalar> scalars =
        min_max_detail::to_vec_scalar(std::forward<Args &&>(args)...);

    if (scalars.size() == 0) {
        return min(std::move(tensors));
    } else if (tensors.size() == 0) {
        Tensor out({1}, scalars[0].type());
        out = min(std::move(scalars));
        return std::move(out);
    }
    Scalar min_s = min(std::move(scalars));
    return min(std::move(tensors), min_s);
}


} // namespace functional
} // namespace nt

#endif
