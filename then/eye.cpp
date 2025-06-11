// #include "../linalg.h"
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

Tensor eye(int64_t n, int64_t b, DType dtype){
	if(b > 0){
		Tensor out = functional::zeros({b, n, n}, dtype);
		out.arr_void().execute_function<WRAP_DTYPES<NumberTypesL>>([&b, &n](auto begin, auto end){
			using value_t = utils::IteratorBaseType_t<decltype(begin)>;
            value_t one = static_cast<value_t>(1);
			for(int64_t k = 0; k < b; ++k){
				for(int64_t i = 0; i < n; ++i){
                    begin += i;
                    *begin = one;
                    begin += (n-i);
				}	
			}


		});
		return std::move(out);
	}
	Tensor out = functional::zeros({n, n}, dtype);
	out.arr_void().execute_function<WRAP_DTYPES<NumberTypesL>>([&n](auto begin, auto end){
		using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t one = static_cast<value_t>(1);
		for(int64_t i = 0; i < n; ++i){
            begin += i;
            *begin = one;
            begin += (n-i);
		}	

	});
	return std::move(out);
}


Tensor eye_like(const Tensor& t){
	utils::throw_exception(t.dims() >= 2, "Expected to make identity matrix with dimensions greater than or equal to 2 but got $", t.dims());
	if(t.dims() == 2){
		return eye(t.shape()[-1], 0, t.dtype);
	}
	auto vec = t.shape().Vec();
	vec[vec.size()-2] = vec.back();
	return eye(t.shape()[-1], t.shape().flatten(0, -3)[0], t.dtype).view(SizeRef(std::move(vec)));
}



}}
