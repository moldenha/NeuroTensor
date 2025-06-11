#include "unique.h"
#include "../tensor_files/combine.h"
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "../../types/Types.h"
#include "../../convert/Convert.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"

namespace nt{
namespace functional{
namespace cpu{


template <typename T>
struct NumericVectorHash {

    std::size_t operator()(const Tensor& vec){
        return vec.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<T> > > >([](auto begin, auto end){
            std::size_t hash = 0;
            for(;begin != end; ++begin){
                if constexpr (std::is_same_v<my_complex<float16_t>, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->real())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->imag())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<my_complex<float>, T>){
                    hash ^= std::hash<float>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<my_complex<double>, T>){
                    hash ^= std::hash<double>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<double>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<float16_t, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<uint_bool_t, T>){
                    hash ^= std::hash<float>{}(*begin ? float(1) : float(0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#ifdef __SIZEOF_INT128__
                else if constexpr(std::is_same_v<uint128_t, T>){
                    hash ^= std::hash<int64_t>{}(convert::convert<int64_t, uint128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<int128_t, T>){
                    hash ^= std::hash<int64_t>{}(convert::convert<int64_t, int128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#endif
                else{
                    hash ^= std::hash<T>{}(*begin) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
            return hash;
        });
    }
};

template<typename T>
struct NumericVectorEqual {
    bool operator()(const Tensor& a, const Tensor& b) const {
        if(a.numel() != b.numel() || a.dtype != b.dtype){return false;}
        if(a.is_null() || b.is_null()){return false;}
        const ArrayVoid& arr_v = b.arr_void();
        return a.arr_void().cexecute_function<DTypeFuncs::type_to_dtype<T> >([&arr_v](auto begin, auto end) -> bool{
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            return arr_v.cexecute_function<DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
                return std::equal(begin, end, second);
            });
        });
    }
};


template<typename T>
struct tensor_hashed{
    const Tensor* a;
    std::size_t hash;
    tensor_hashed(const Tensor* a_) : a(a_) {
        hash = a_->arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DTypeFuncs::type_to_dtype<T> > > >([](auto begin, auto end){
            std::size_t hash = 0;
            for(;begin != end; ++begin){
                if constexpr (std::is_same_v<my_complex<float16_t>, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->real())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(begin->imag())) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<my_complex<float>, T>){
                    hash ^= std::hash<float>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<float>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<my_complex<double>, T>){
                    hash ^= std::hash<double>{}(begin->real()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    hash ^= std::hash<double>{}(begin->imag()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr (std::is_same_v<float16_t, T>){
                    hash ^= std::hash<float>{}(_NT_FLOAT16_TO_FLOAT32_(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<uint_bool_t, T>){
                    hash ^= std::hash<float>{}(*begin ? float(1) : float(0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#ifdef __SIZEOF_INT128__
                else if constexpr(std::is_same_v<uint128_t, T>){
                    hash ^= std::hash<int64_t>{}(convert::convert<int64_t, uint128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                else if constexpr(std::is_same_v<int128_t, T>){
                    hash ^= std::hash<int64_t>{}(convert::convert<int64_t, int128_t>(*begin)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
#endif
                else{
                    hash ^= std::hash<T>{}(*begin) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
            return hash;
        });

    }
};

template <typename T>
struct HashedTensorHash {
    std::size_t operator()(const tensor_hashed<T>& vec) const { return vec.hash;}
};

template<typename T>
struct  HashedTensorEqual {
    bool operator()(const tensor_hashed<T>& h_a, const tensor_hashed<T>& h_b) const {
        const Tensor& a = *h_a.a;
        const Tensor& b = *h_b.a;
        if(a.numel() != b.numel() || a.dtype != b.dtype){return false;}
        if(a.is_null() || b.is_null()){return false;}
        const ArrayVoid& arr_v = b.arr_void();
        return a.arr_void().cexecute_function<DTypeFuncs::type_to_dtype<T> >([&arr_v](auto begin, auto end) -> bool{
            using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            return arr_v.cexecute_function<DTypeFuncs::type_to_dtype<value_t>>([&begin, &end](auto second, auto s_end) -> bool{
                return std::equal(begin, end, second);
            });
        });
    }
};

Tensor _unique(Tensor input, int64_t dim, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "unique function must return indices or the sorted tensor, or both, got none");
    input = input.transpose(-1, dim).contiguous();
    int64_t last_dim = input.shape().back();
    Tensor splits = input.split_axis(-2);
    Tensor* s_begin = reinterpret_cast<Tensor*>(splits.data_ptr());
    Tensor* s_end = reinterpret_cast<Tensor*>(splits.data_ptr_end());
    return input.arr_void().execute_function<WRAP_DTYPES<NumberTypesL> >(
    [&s_begin, &s_end, &last_dim, &return_sorted, &return_indices](auto begin, auto end) -> Tensor {
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        std::unordered_map<tensor_hashed<value_t>, int64_t, 
            HashedTensorHash<value_t>, HashedTensorEqual<value_t>> unique_map; //tensor and its indice
        int64_t counter = 0;
        unique_map[tensor_hashed<value_t>(s_begin)] = counter;
        ++counter;
        ++s_begin;
        for(;s_begin != s_end; ++s_begin, ++counter){
            tensor_hashed<value_t> check(s_begin);
            if (unique_map.find(check) == unique_map.end()) {
                unique_map[check] = counter;
            } 
        }
        if(!return_indices){
            Tensor output = Tensor::makeNullTensorArray(static_cast<long long>(unique_map.size()));
            Tensor* o_begin = reinterpret_cast<Tensor*>(output.data_ptr());
            Tensor* o_end = o_begin + output.numel();
            for(const auto& [tensor, indice] : unique_map){
                *o_begin = *tensor.a;
                ++o_begin;
            }
            Tensor out = cat_unordered(output);
            return out.view(-1, last_dim);
        }
        if(!return_sorted){
            Tensor output_indices({static_cast<long long>(unique_map.size())}, DType::int64);
            int64_t* i_begin = reinterpret_cast<int64_t*>(output_indices.data_ptr());
            for(const auto& [tensor, indice] : unique_map){
                *i_begin = indice;
                ++i_begin;
            }
            return std::move(output_indices);
 
        }
        Tensor output = Tensor::makeNullTensorArray(static_cast<long long>(unique_map.size()));
        Tensor output_indices({static_cast<long long>(unique_map.size())}, DType::int64);
        Tensor* o_begin = reinterpret_cast<Tensor*>(output.data_ptr());
        Tensor* o_end = o_begin + output.numel();
        int64_t* i_begin = reinterpret_cast<int64_t*>(output_indices.data_ptr());
        for(const auto& [tensor, indice] : unique_map){
            *i_begin = indice;
            *o_begin = *tensor.a;
            ++i_begin;
            ++o_begin;
        }
        Tensor out = cat_unordered(output);
        return list(out.view(-1, last_dim), output_indices);
    });
}

}
}
}
