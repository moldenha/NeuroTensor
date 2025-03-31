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

//TODO: add nuclear norm
Tensor norm(const Tensor& A, std::variant<std::nullptr_t, std::string, int64_t> ord, utils::optional_list dim, bool keepdim){
	if(ord.index() == 0){
		return functional::sqrt(std::pow(A, 2).sum(dim, keepdim));
	}
	else if(ord.index() == 1){
		std::string name = std::get<1>(ord);
		utils::throw_exception(name == "fro",
				"Order strings accepted are fro but got $", name);
		if(name == "fro"){
			return functional::sqrt(std::pow(A, 2).sum(dim, keepdim));
		}
	}
	int64_t ord_i = std::get<2>(ord);
	if(ord_i == std::numeric_limits<int64_t>::infinity()){
		return functional::abs(A).max(dim).values;
	}
	if(ord_i == -std::numeric_limits<int64_t>::infinity()){
		return functional::abs(A).min(dim).values;
	}
	utils::throw_exception(ord_i > 0, "Order must be fro or positive integer");
	Tensor a_out = std::pow(functional::abs(A), ord_i).sum(dim);
	return std::pow(a_out, (1.0 / (double)ord_i));
}


}
}
