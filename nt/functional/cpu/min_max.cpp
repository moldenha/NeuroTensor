#include "min_max.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace nt{
namespace functional{
namespace cpu{

void _clamp(ArrayVoid& a, Scalar min, Scalar max){
    a.execute_function<WRAP_DTYPES<NumberTypesL>>([&min, &max](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t n_min = min.to<value_t>();
        value_t n_max = max.to<value_t>();
        for(;begin != end; ++begin){
            *begin = std::clamp(*begin, n_min, n_max);
        }
    });
}



bool all_contiguous(std::vector<ArrayVoid>& arrvds){
    for(const auto& arr : arrvds){
        utils::THROW_EXCEPTION(arr.is_contiguous(), "INTERNAL LOGIC ERROR: Expected all array voids to be contiguous in cpu processing");
    }
    return true;
}

#define _NT_CPU_CUR_MIN_(a, b) ((a < b) ? a : b)

void _min(ArrayVoid& out, std::vector<ArrayVoid>& arrvds){
    all_contiguous(arrvds);

    out.execute_function<WRAP_DTYPES<NumberTypesL> >([&arrvds](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        size_t numel = arrvds.size();
        value_t* iterators[numel];
        size_t i;
        for(i = 0; i < numel; ++i){
            iterators[i] = reinterpret_cast<value_t*>(arrvds[i].data_ptr());
        }
        for(;begin != end; ++begin){
            for(i = 0; i < numel; ++i){
                *begin = _NT_CPU_CUR_MIN_(*begin, *iterators[i]);
                ++iterators[i];
            }
        }
    });
}

#undef _NT_CPU_CUR_MIN_

#define _NT_CPU_CUR_MAX_(a, b) ((a < b) ? b : a)

void _max(ArrayVoid& out, std::vector<ArrayVoid>& arrvds){
    all_contiguous(arrvds);

    out.execute_function<WRAP_DTYPES<NumberTypesL> >([&arrvds](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        size_t numel = arrvds.size();
        value_t* iterators[numel];
        size_t i;
        for(i = 0; i < numel; ++i){
            iterators[i] = reinterpret_cast<value_t*>(arrvds[i].data_ptr());
        }
        for(;begin != end; ++begin){
            for(i = 0; i < numel; ++i){
                *begin = _NT_CPU_CUR_MAX_(*begin, *iterators[i]);
                ++iterators[i];
            }
        }
    });
}
#undef _NT_CPU_CUR_MAX_


}
}
}
