#include "../utils/utils.h"
#include "../Tensor.h"
#include "../functional/functional.h"
#include <limits>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	#include <tbb/parallel_reduce.h>
	#include <thread>
    #include <tbb/concurrent_vector.h>
	/* #include "../mp/MP.hpp" */
	/* #include "../mp/Pool.hpp" */
#endif

namespace nt{
namespace linalg{

Tensor indp_rows(const Tensor& _a, const Tensor& _b){
	utils::throw_exception(_a.dims() >= 2 && _b.dims() >= 2, "Expected to find independent rows from tensors that dont have more than 1 dim got $ and $", _a.dims(), _b.dims());

    //should ensure all rows are unique coming in
    Tensor a = _a.flatten(0, -2);
    Tensor b = _b.flatten(0, -2);
    //if(a.shape()[-2] < b.shape()[-2]){std::swap(a, b);}
    //a should now have more rows than b
    std::vector<bool> independent(a.shape()[-2], true);
    Tensor a_split = a.split_axis(-2);
    Tensor b_split = b.split_axis(-2);
    
    Tensor* a_begin = reinterpret_cast<Tensor*>(a_split.data_ptr());
    Tensor* a_end = a_begin + a_split.numel();
#ifdef USE_PARALLEL
    tbb::parallel_for(nt::utils::calculateGrainSize1D(a_split.numel()),
    [&](const tbb::blocked_range<int64_t> &range){
    Tensor* b_begin = reinterpret_cast<Tensor*>(b_split.data_ptr());
    Tensor* b_end = b_begin + b_split.numel();
    auto b_cpy = b_begin;
    for(int64_t k = range.begin(); k != range.end(); ++k){
        for(;b_begin != b_end; ++b_begin){
            if(functional::all(functional::abs(*b_begin) == functional::abs(a_begin[k]))){
                independent[k] = false;
                break;
            }
        }
        b_begin = b_cpy;
    }
    });
#else
    Tensor* b_begin = reinterpret_cast<Tensor*>(b_split.data_ptr());
    Tensor* b_end = b_begin + b_split.numel();
    auto b_cpy = b_begin;
    int64_t k = 0;
    for(;a_begin != a_end; ++a_begin, ++k){
        for(;b_begin != b_end; ++b_begin){
           if(functional::all(*b_begin == *a_begin)){
                independent[k] = false;
                break;
            }
        }
    }
#endif
    a_begin = reinterpret_cast<Tensor*>(a_split.data_ptr());
    std::vector<Tensor> out;
    int64_t amt = std::count(independent.begin(), independent.end(), true);
    if(amt == 0){return Tensor::Null();}
    out.reserve(amt);
    for(int64_t k = 0; k < independent.size(); ++k){
        if(independent[k]){
            out.push_back(a_begin[k]);
        }
    }
    return functional::stack(out);
}

Tensor indp_cols(const Tensor& _a, const Tensor& _b){
    Tensor a = _a.transpose(-1, -2);
    Tensor b = _b.transpose(-1, -2);
    Tensor out = indp_rows(a, b);
    if(out.is_null()){return out;}
    return out.transpose(-1, -2);
}




}}
