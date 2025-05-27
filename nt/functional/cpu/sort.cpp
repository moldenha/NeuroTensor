#include "sort.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <algorithm>
#include "../../mp/Threading.h"

namespace nt{
namespace functional{
namespace cpu{


template<typename Iterator>
inline bool _nt_sort_descending_(const int64_t& a, const int64_t& b, const Iterator& data){
    return data[a] > data[b];
}

template<typename Iterator>
inline bool _nt_sort_ascending_(const int64_t& a, const int64_t& b, const Iterator& data){
    return data[a] < data[b];
}

void _sort_vals_only_(ArrayVoid& values, const bool& descending, const int64_t& dim_size){
    if(values.dtype == DType::TensorObj){
        throw std::invalid_argument("_sort_vals_only_ is not designated to handle tensor dtype, try _sort_vals_dtype_tensor_only_");
    }
    if(!values.is_contiguous()){
        throw std::invalid_argument("values to cpu::_sort_vals_only_ must be contiguous");
    }
    values.execute_function<WRAP_DTYPES<NumberTypesL> >(
    [&descending, &dim_size](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        using iterator_t = decltype(begin);
        if constexpr (std::is_pointer_v<iterator_t>){ // because values were cloned should all be contiguous, but this ensures it
        int64_t total = (end - begin) / dim_size;
#ifdef USE_PARALLEL
        if(descending){
            threading::preferential_parallel_for(
                threading::block_ranges<1>(0, total),
                [&](threading::blocked_range<1> block){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = begin + (block.begin[0] * dim_size);
                    auto i_end = begin + (block.end[0] * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                        std::sort(i_begin, i_begin + dim_size, std::greater<value_t>());
                    }
            });
        }else{
           threading::preferential_parallel_for(
                threading::block_ranges<1>(0, total),
                [&](threading::blocked_range<1> block){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = begin + (block.begin[0] * dim_size);
                    auto i_end = begin + (block.end[0] * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                        std::sort(i_begin, i_begin + dim_size, std::less<value_t>());
                    }
            });

        }
#else
        if(descending){
            for(;begin != _end; begin += dim_size){
                std::sort(begin, begin + dim_size, std::greater<value_t>());
            }
        }else{
            for(;indices_begin != indices_end; indices_begin += dim_size){
                std::sort(indices_begin, indices_begin + dim_size, std::less<value_t>());
            }
        }
#endif
    
    }});
    
}



void _sort_(ArrayVoid& values, int64_t* indices_begin, int64_t* indices_end, const bool& descending, const int64_t& dim_size){
    if(values.dtype == DType::TensorObj){
        throw std::invalid_argument("_sort_ is not designated to handle tensor dtype, try _sort_tensor_");
    }
    if(!values.is_contiguous()){
        throw std::invalid_argument("values to cpu::_sort_ must be contiguous");
    }
    values.execute_function<WRAP_DTYPES<NumberTypesL> >(
    [&descending, &dim_size, &indices_begin](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        using iterator_t = decltype(begin);
        if constexpr (std::is_pointer_v<iterator_t>){ // because values were cloned should all be contiguous, but this ensures it
        int64_t total = (end - begin) / dim_size;
#ifdef USE_PARALLEL
        if(descending){
            threading::preferential_parallel_for(
                threading::block_ranges<1>(0, total),
                [&](threading::blocked_range<1> block){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = indices_begin + (block.begin[0] * dim_size);
                    auto i_end = indices_begin + (block.end[0] * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                         std::sort(i_begin, i_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                            return _nt_sort_descending_<iterator_t>(a, b, begin);    
                         });
                    }
            });
        }else{
           threading::preferential_parallel_for(
                threading::block_ranges<1>(0, total),
                [&](threading::blocked_range<1> block){
                    // auto s_begin = begin + (range.begin() * dim_size);
                    // auto s_end = begin + (range.end() * dim_size);
                    auto i_begin = indices_begin + (block.begin[0] * dim_size);
                    auto i_end = indices_begin + (block.end[0] * dim_size);
                    for(;i_begin < i_end; i_begin += dim_size){
                        std::sort(i_begin, i_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                            return _nt_sort_ascending_<iterator_t>(a, b, begin);    
                         });
                    }
            });

        }
#else
        if(descending){
            for(;indices_begin != indices_end; indices_begin += dim_size){
                std::sort(indices_begin, indices_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                    return _nt_sort_descending_<iterator_t>(a, b, begin);    
                });
            }
        }else{
            for(;indices_begin != indices_end; indices_begin += dim_size){
                std::sort(indices_begin, indices_begin + dim_size, [&begin](const int64_t& a, const int64_t& b){
                    return _nt_sort_ascending_<iterator_t>(a, b, begin);
                });
            }
        }
#endif
    
    }});

}

}
}
}
