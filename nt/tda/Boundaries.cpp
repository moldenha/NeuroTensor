#include "Boundaries.h"
#include "../convert/std_convert.h"
#include "../functional/functional.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../dtype/ArrayVoid.hpp"

namespace nt {
namespace tda {

template <typename T> struct NumericVectorHash {

    std::size_t operator()(const Tensor &vec) const {
        return vec.arr_void()
            .cexecute_function<WRAP_DTYPES<
                DTypeEnum<DTypeFuncs::type_to_dtype<T>>>>([](auto begin,
                                                             auto end) {
                std::size_t hash = 0;
                for (; begin != end; ++begin) {
                    if constexpr (std::is_same_v<my_complex<float16_t>, T>) {
                        hash ^= std::hash<float>{}(
                                    _NT_FLOAT16_TO_FLOAT32_(begin->real())) +
                                0x9e3779b9 + (hash << 6) + (hash >> 2);
                        hash ^= std::hash<float>{}(
                                    _NT_FLOAT16_TO_FLOAT32_(begin->imag())) +
                                0x9e3779b9 + (hash << 6) + (hash >> 2);
                    } else if constexpr (std::is_same_v<my_complex<float>, T>) {
                        hash ^= std::hash<float>{}(begin->real()) + 0x9e3779b9 +
                                (hash << 6) + (hash >> 2);
                        hash ^= std::hash<float>{}(begin->imag()) + 0x9e3779b9 +
                                (hash << 6) + (hash >> 2);
                    } else if constexpr (std::is_same_v<my_complex<double>,
                                                        T>) {
                        hash ^= std::hash<double>{}(begin->real()) +
                                0x9e3779b9 + (hash << 6) + (hash >> 2);
                        hash ^= std::hash<double>{}(begin->imag()) +
                                0x9e3779b9 + (hash << 6) + (hash >> 2);
                    } else if constexpr (std::is_same_v<float16_t, T>) {
                        hash ^= std::hash<float>{}(
                                    _NT_FLOAT16_TO_FLOAT32_(*begin)) +
                                0x9e3779b9 + (hash << 6) + (hash >> 2);
                    } else if constexpr (std::is_same_v<uint_bool_t, T>) {
                        hash ^=
                            std::hash<float>{}(*begin ? float(1) : float(0)) +
                            0x9e3779b9 + (hash << 6) + (hash >> 2);
                    }
#ifdef __SIZEOF_INT128__
                    else if constexpr (std::is_same_v<uint128_t, T>) {
                        hash ^=
                            std::hash<int64_t>{}(
                                convert::convert<int64_t, uint128_t>(*begin)) +
                            0x9e3779b9 + (hash << 6) + (hash >> 2);
                    } else if constexpr (std::is_same_v<int128_t, T>) {
                        hash ^=
                            std::hash<int64_t>{}(
                                convert::convert<int64_t, int128_t>(*begin)) +
                            0x9e3779b9 + (hash << 6) + (hash >> 2);
                    }
#endif
                    else {
                        hash ^= std::hash<T>{}(*begin) + 0x9e3779b9 +
                                (hash << 6) + (hash >> 2);
                    }
                }
                return hash;
            });
    }
};

template <typename T> struct NumericVectorEqual {
    bool operator()(const Tensor &a, const Tensor &b) const {
        if (a.is_null() || b.is_null()) {
            return false;
        }
        if (a.numel() != b.numel() || a.dtype != b.dtype) {
            return false;
        }
        const ArrayVoid &arr_v = b.arr_void();
        return a.arr_void().cexecute_function<DTypeFuncs::type_to_dtype<T>>(
            [&arr_v](auto begin, auto end) -> bool {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                return arr_v
                    .cexecute_function<DTypeFuncs::type_to_dtype<value_t>>(
                        [&begin, &end](auto second, auto s_end) -> bool {
                            return std::equal(begin, end, second);
                        });
            });
    }
};

// these only take int64_t indexes
struct tensor_index_skip {
    Tensor a;
    int64_t skip;
    std::size_t hash;
    tensor_index_skip(const Tensor &a_, int64_t skip_) : a(a_), skip(skip_) {
        hash = compute_hash(); // Precompute hash on construction
    }

    std::size_t compute_hash() const {
        return a.arr_void()
            .cexecute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>(
                [&](auto begin, auto end) {
                    std::size_t hash = 0;
                    int64_t c = 0;
                    for (; begin != end; ++begin, ++c) {
                        if (c == skip)
                            continue;
                        hash ^= std::hash<int64_t>{}(*begin) + 0x9e3779b9 +
                                (hash << 6) + (hash >> 2);
                    }
                    return hash;
                });
    }
};

struct TensorSkipHash {
    std::size_t operator()(const tensor_index_skip &v) const { return v.hash; }
};

template <typename T = int64_t> struct TensorSkipEqual {
    bool operator()(const Tensor &a, const Tensor &b) const {
        if (a.is_null() || b.is_null()) {
            return false;
        }
        if (a.numel() != b.numel() || a.dtype != b.dtype) {
            return false;
        }
        const ArrayVoid &arr_v = b.arr_void();
        return a.arr_void().cexecute_function<DTypeFuncs::type_to_dtype<T>>(
            [&arr_v](auto begin, auto end) -> bool {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                return arr_v
                    .cexecute_function<DTypeFuncs::type_to_dtype<value_t>>(
                        [&begin, &end](auto second, auto s_end) -> bool {
                            return std::equal(begin, end, second);
                        });
            });
    }
    bool operator()(const tensor_index_skip &a,
                    const tensor_index_skip &b) const {
        const ArrayVoid &arr_v = b.a.arr_void();
        return a.a.arr_void().cexecute_function<DTypeFuncs::type_to_dtype<T>>(
            [&arr_v, &a, &b](auto begin, auto end) -> bool {
                using value_t = utils::IteratorBaseType_t<decltype(begin)>;
                return arr_v
                    .cexecute_function<DTypeFuncs::type_to_dtype<value_t>>(
                        [&begin, &end, &a, &b](auto second,
                                               auto s_end) -> bool {
                            int64_t c = 0;
                            for (; begin != end; ++begin, ++second, ++c) {
                                if (c == a.skip)
                                    ++begin;
                                if (c == b.skip)
                                    ++second;
                                if (begin == end || second == s_end)
                                    break;
                                if (*begin != *second)
                                    return false;
                            }
                            return true;
                        });
            });
    }
};

SparseTensor compute_boundary_matrix_index(const Tensor &s_kp1,
                                           const Tensor &s_k) {
    utils::throw_exception(
        s_kp1.dims() == 2,
        "Expected to get simplicies of dims 2, corresponding to batches, "
        "verticies, point dim but got $ for simplex complex k+1",
        s_kp1.dims());
    utils::throw_exception(
        s_k.dims() == 2,
        "Expected to get simplicies of dims 2, corresponding to batches, "
        "verticies, point dim but got $ for simplex complex k",
        s_k.dims());
    utils::throw_exception(
        s_kp1.dtype == DType::int64,
        "Expected simplicies (k+1) dtype to be int64 but got $", s_kp1.dtype);
    utils::throw_exception(
        s_k.dtype == DType::int64,
        "Expected simplicies (k) dtype to be int64 but got $", s_k.dtype);
    const int64_t &B_kp1 = s_kp1.shape()[0]; // Batches
    const int64_t Kp1 = s_kp1.shape()[1];    // K+1 verticies

    const int64_t &B_k = s_k.shape()[0]; // Batches
    const int64_t K = s_k.shape()[1];    // K verticies

    utils::throw_exception(
        K == (Kp1 - 1), "Expected K ($) to be one less than K+1 ($) ($ -> $)",
        K, Kp1, s_kp1.shape(), s_k.shape());

    // Tensor faces = functional::combinations(functional::arange(Np1,
    // DType::int64), Np1-1).contiguous().split_axis(0); Tensor* f_begin =
    // reinterpret_cast<Tensor*>(faces.data_ptr()); Tensor* f_end =
    // reinterpret_cast<Tensor*>(faces.data_ptr_end()); Tensor B_dim =
    // functional::arange(B, DType::int64).view(B, 1).repeat_(-1,
    // D*(Np1-1)).flatten(0, -1).contiguous(); Tensor point_dim =
    // functional::arange(D, DType::int64).repeat_((Np1-1) * B).flatten(0,
    // -1).contiguous();

    std::unordered_map<tensor_index_skip,
                       std::pair<std::vector<int64_t>, std::vector<int8_t>>,
                       TensorSkipHash, TensorSkipEqual<int64_t>>
        simplex_map;
    // std::unordered_map<uint64_t, std::pair<int64_t, int64_t>>
    // Tensor face_map = nt:Tensor::makeNullTensorArray(Kp1);
    // Tensor* f_begin
    for (int64_t i = 0; i < Kp1; ++i) {
        // now get all faces
        Tensor split_faces = s_kp1.split_axis(0);
        int64_t counter = 0;
        Tensor *split_begin =
            reinterpret_cast<Tensor *>(split_faces.data_ptr());
        Tensor *split_end =
            reinterpret_cast<Tensor *>(split_faces.data_ptr_end());
        int8_t boundary = (i % 2 == 0) ? 1 : -1;
        for (; split_begin != split_end; ++split_begin, ++counter) {
            // const int64_t* arr_sub = reinterpret_cast<const
            // int64_t*>(split_begin->data_ptr());
            tensor_index_skip key(*split_begin, i);
            auto [it, inserted] =
                simplex_map.insert({key,
                                    {std::vector<int64_t>({counter}),
                                     std::vector<int8_t>({boundary})}});
            if (!inserted) {
                auto &pair = it->second;
                pair.first.push_back(counter);
                pair.second.push_back(boundary);
            }
        }
    }

    // each collumn refers to a simplex from s_kp1, and each row refers to a
    // simplex from s_k So, x -> from B_k, and y is from the map

    std::vector<int64_t> x;
    x.reserve(simplex_map.size());
    std::vector<int64_t> y;
    y.reserve(simplex_map.size());
    std::vector<int8_t> boundaries;
    boundaries.reserve(simplex_map.size());
    Tensor sk_split = s_k.split_axis(0);
    Tensor *sk_begin = reinterpret_cast<Tensor *>(sk_split.data_ptr());
    Tensor *sk_end = reinterpret_cast<Tensor *>(sk_split.data_ptr_end());
    int64_t x_counter = 0;
    for (; sk_begin != sk_end; ++sk_begin, ++x_counter) {
        // const int64_t* arr = reinterpret_cast<const
        // int64_t*>(sk_begin->data_ptr()); std::vector<int64_t> key(arr, arr +
        // K); nt_index_holder key = to_index_holder(reinterpret_cast<const
        // int64_t*>(sk_begin->data_ptr()), K);
        tensor_index_skip cur_key(*sk_begin, -1);
        if (simplex_map.find(cur_key) == simplex_map.end()) {
            continue;
        }
        const auto &pair = simplex_map[cur_key];
        y.insert(y.end(), pair.first.cbegin(), pair.first.cend());
        boundaries.insert(boundaries.end(), pair.second.cbegin(),
                          pair.second.cend());
        x.insert(x.end(), pair.first.size(), x_counter);
    }

    Tensor X = functional::vector_to_tensor(x);
    Tensor Y = functional::vector_to_tensor(y);
    Tensor Boundaries = functional::vector_to_tensor(boundaries);
    return SparseTensor(functional::list(X, Y), Boundaries, {B_k, B_kp1},
                        DType::int8, 0);
}

SparseTensor compute_boundary_matrix(const Tensor &s_kp1, const Tensor &s_k) {
    utils::throw_exception(
        s_kp1.dims() == s_k.dims(),
        "Expected simplicies to have same dimensions but got $ and $",
        s_kp1.dims(), s_k.dims());
    if (s_kp1.dims() == 2) {
        return compute_boundary_matrix_index(s_kp1, s_k);
    }
    utils::throw_exception(
        s_kp1.dims() == 3,
        "Expected to get simplicies of dims 3, corresponding to batches, "
        "verticies, point dim but got $ for simplex complex k+1",
        s_kp1.dims());
    utils::throw_exception(
        s_k.dims() == 3,
        "Expected to get simplicies of dims 3, corresponding to batches, "
        "verticies, point dim but got $ for simplex complex k",
        s_k.dims());
    utils::throw_exception(
        s_kp1.dtype == DType::int64,
        "Expected simplicies (k+1) dtype to be int64 but got $", s_kp1.dtype);
    utils::throw_exception(
        s_k.dtype == DType::int64,
        "Expected simplicies (k) dtype to be int64 but got $", s_k.dtype);
    const int64_t &B_kp1 = s_kp1.shape()[0]; // Batches
    const int64_t Kp1 = s_kp1.shape()[1];    // K+1 verticies
    const int64_t &D = s_kp1.shape()[2];     // Dims

    const int64_t &B_k = s_k.shape()[0]; // Batches
    const int64_t K = s_k.shape()[1];    // K verticies
    const int64_t &D_k = s_k.shape()[2]; // Dims

    utils::throw_exception(
        K == (Kp1 - 1), "Expected K ($) to be one less than K+1 ($) ($ -> $)",
        K, Kp1, s_kp1.shape(), s_k.shape());
    utils::throw_exception(
        D == D_k, "Expected dims of points to be equal but got $ and $", D,
        D_k);

    // Tensor faces = functional::combinations(functional::arange(Np1,
    // DType::int64), Np1-1).contiguous().split_axis(0); Tensor* f_begin =
    // reinterpret_cast<Tensor*>(faces.data_ptr()); Tensor* f_end =
    // reinterpret_cast<Tensor*>(faces.data_ptr_end()); Tensor B_dim =
    // functional::arange(B, DType::int64).view(B, 1).repeat_(-1,
    // D*(Np1-1)).flatten(0, -1).contiguous(); Tensor point_dim =
    // functional::arange(D, DType::int64).repeat_((Np1-1) * B).flatten(0,
    // -1).contiguous();

    std::unordered_map<Tensor,
                       std::pair<std::vector<int64_t>, std::vector<int8_t>>,
                       NumericVectorHash<int64_t>, NumericVectorEqual<int64_t>>
        simplex_map;
    for (int64_t i = 0; i < Kp1; ++i) {
        // this is the faces operator Simplex(k+1) -> Simplex(k)
        Tensor parted_simplicies = s_kp1.index_except(1, i);
        // now get all faces
        Tensor split_faces = parted_simplicies.split_axis(0);
        int64_t counter = 0;
        Tensor *split_begin =
            reinterpret_cast<Tensor *>(split_faces.data_ptr());
        Tensor *split_end =
            reinterpret_cast<Tensor *>(split_faces.data_ptr_end());
        int8_t boundary = (i % 2 == 0) ? 1 : -1;
        for (; split_begin != split_end; ++split_begin, ++counter) {
            auto [it, inserted] =
                simplex_map.insert({*split_begin,
                                    {std::vector<int64_t>({counter}),
                                     std::vector<int8_t>({boundary})}});
            if (!inserted) {
                auto &pair = it->second;
                pair.first.push_back(counter);
                pair.second.push_back(boundary);
            }
        }
    }

    // each collumn refers to a simplex from s_kp1, and each row refers to a
    // simplex from s_k So, x -> from B_k, and y is from the map

    std::vector<int64_t> x;
    x.reserve(simplex_map.size());
    std::vector<int64_t> y;
    y.reserve(simplex_map.size());
    std::vector<int8_t> boundaries;
    boundaries.reserve(simplex_map.size());
    Tensor sk_split = s_k.split_axis(0);
    Tensor *sk_begin = reinterpret_cast<Tensor *>(sk_split.data_ptr());
    Tensor *sk_end = reinterpret_cast<Tensor *>(sk_split.data_ptr_end());
    int64_t x_counter = 0;
    for (; sk_begin != sk_end; ++sk_begin, ++x_counter) {
        if (simplex_map.find(*sk_begin) == simplex_map.end()) {
            continue;
        }
        const auto &pair = simplex_map[*sk_begin];
        y.insert(y.end(), pair.first.cbegin(), pair.first.cend());
        boundaries.insert(boundaries.end(), pair.second.cbegin(),
                          pair.second.cend());
        x.insert(x.end(), pair.first.size(), x_counter);
    }

    Tensor X = functional::vector_to_tensor(x);
    Tensor Y = functional::vector_to_tensor(y);
    Tensor Boundaries = functional::vector_to_tensor(boundaries);
    return SparseTensor(functional::list(X, Y), Boundaries, {B_k, B_kp1},
                        DType::int8, 0);
}

} // namespace tda
} // namespace nt
