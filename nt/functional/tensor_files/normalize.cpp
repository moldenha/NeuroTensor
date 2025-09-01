#include "normalize.h"
#include "../cpu/normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"
#include <cmath>
#include "exceptions.hpp"
#include "activation_functions.h"

namespace nt{
namespace functional{

Tensor& xavier_uniform_(Tensor& tensor){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(tensor);
    utils::throw_exception(tensor.is_mutable(), "Can only perform xavier_uniform_ on a mutable tensor");
    utils::throw_exception(tensor.dims() >= 2, "For xavier uniform the dimensions of the tensor must be greater than or equal to 2");
    int64_t fan_in = tensor.shape()[-1]; //switch to [1] maybe
    int64_t fan_out = tensor.shape()[-2]; //switch to [0] maybe
    double bound = std::sqrt(6.0 / (double)(fan_in + fan_out));
    cpu::xavier_uniform_(tensor.arr_void(), bound);
    return tensor;
}


Tensor var(const Tensor& x, utils::optional_list dim, int64_t correction, bool keepdim){
	Tensor mean = x.mean(dim, true);
	Tensor squared_diff = pow((x - mean), 2);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	Tensor variance = squared_diff.sum(dim, keepdim) / (N - correction);
	return std::move(variance);
}

Tensor dvar(const Tensor& dx, const Tensor& x, utils::optional_list dim, int64_t correction){
	//takes both the gradient, and the input given to the variance function
	Tensor mean = x.mean(dim, true);
	int64_t N = 0;
	if(!dim){
		N = x.numel();
	}else{
		N = 1;
		for(const auto& ele : dim){
			N *= x.shape()[ele];
		}
	}
	return (2 / (N - correction)) * (x - mean);
}


}
}
