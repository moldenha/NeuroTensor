#include "unique.h"
#include "../tensor_files/combine.h"
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "../../types/Types.h"
#include "../../types/hash.h"
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
                hash ^= nt::types::hash<T>{}(*begin) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        });
    }
};

template<typename T>
struct NumericVectorEqual {
    bool operator()(const Tensor& a, const Tensor& b) const {
        if(a.numel() != b.numel() || a.dtype() != b.dtype()){return false;}
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
                hash ^= nt::types::hash<T>{}(*begin) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
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
        if(a.numel() != b.numel() || a.dtype() != b.dtype()){return false;}
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

Tensor _unique_vals_only(const Tensor& input, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "unique function must return indices or the sorted tensor, or both, got none");
    utils::throw_exception(input.dtype() != DType::TensorObj, "Error: Unique does not support Tensor dtype, please implement each tensor seperately");
    Tensor indices = input.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >(
    [&return_sorted, &return_indices](auto begin, auto end) -> Tensor {
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        nt::types::unordered_map<value_t, int64_t> unique_map;
        int64_t counter = 0;
        unique_map[*begin] = counter;
        ++begin;
        ++counter;
        for(;begin != end; ++begin, ++counter){
            if(unique_map.find(*begin) == unique_map.end()){
                unique_map[*begin] = counter;
            }
        }
        Tensor out({static_cast<int64_t>(unique_map.size())}, DType::int64);
        int64_t* o_begin = reinterpret_cast<int64_t*>(out.data_ptr());
        int64_t* o_end = reinterpret_cast<int64_t*>(out.data_ptr_end());
        for(auto iter = unique_map.cbegin(); iter != unique_map.cend(); ++iter, ++o_begin){
            *o_begin = iter->second;
        }
        std::sort(reinterpret_cast<int64_t*>(out.data_ptr()), o_end);
        return std::move(out);
    });
    
    if(!return_sorted){return std::move(indices);}
    if(!return_indices){return input.flatten(0, -1)[indices];}
    return list(input.flatten(0, -1)[indices], indices);
    
}

Tensor _unique(const Tensor& _input, int64_t dim, bool return_sorted, bool return_indices){
    utils::throw_exception(return_sorted || return_indices, "unique function must return indices or the sorted tensor, or both, got none");
    utils::throw_exception(_input.dtype() != DType::TensorObj, "Error: Unique does not support Tensor dtype, please implement each tensor seperately");
    Tensor input = _input.transpose(-1, dim).contiguous();
    int64_t last_dim = input.shape().back();
    Tensor splits = input.split_axis(-2);
    Tensor* s_begin = reinterpret_cast<Tensor*>(splits.data_ptr());
    Tensor* s_end = reinterpret_cast<Tensor*>(splits.data_ptr_end());
    Tensor indices = input.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>> >(
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
        Tensor out({static_cast<int64_t>(unique_map.size())}, DType::int64);
        int64_t* o_begin = reinterpret_cast<int64_t*>(out.data_ptr());
        int64_t* o_end = reinterpret_cast<int64_t*>(out.data_ptr_end());
        for(auto iter = unique_map.cbegin(); iter != unique_map.cend(); ++iter, ++o_begin){
            *o_begin = iter->second;
        }
        std::sort(reinterpret_cast<int64_t*>(out.data_ptr()), o_end);
        return std::move(out);
    });
    
    if(!return_sorted){return std::move(indices);}
    // Tensor sorted__ = Tensor::makeNullTensorArray(indices.numel());
    // Tensor* o_begin = reinterpret_cast<Tensor*>(sorted__.data_ptr());
    // Tensor* o_end = reinterpret_cast<Tensor*>(sorted__.data_ptr());
    // int64_t* i_begin = reinterpret_cast<int64_t*>(indices.data_ptr());
    // int64_t* i_end = reinterpret_cast<int64_t*>(indices.data_ptr_end());
    // s_begin = reinterpret_cast<Tensor*>(splits.data_ptr());
    // for(;i_begin != i_end; ++i_begin, ++o_begin){
    //     *o_begin = s_begin[*i_begin];
    // }
    Tensor sorted = cat_unordered(splits[indices]).view(-1, last_dim);
    if(!return_indices){return std::move(sorted);}
    return list(std::move(sorted), std::move(indices));


}

}
}
}
