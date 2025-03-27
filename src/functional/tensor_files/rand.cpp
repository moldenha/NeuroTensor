#include "rand.h"
#include "../cpu/rand.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"

namespace nt{
namespace functional{

Tensor randint(Scalar lower, Scalar upper, SizeRef s, DType dt){
    Tensor output(std::move(s), dt);
    cpu::randint_(output.arr_void(), upper, lower);
    return std::move(output);
}

Tensor rand(Scalar lower, Scalar upper, SizeRef s, DType dt){
    Tensor output(std::move(s), dt);
    cpu::rand_(output.arr_void(), upper, lower);
    return std::move(output);
}

Tensor randbools(SizeRef s, double p){
	utils::throw_exception(p >= 0 || p <= 1, "Expected percentage p to be in [0, 1] but got $", p);
	if(p == 1){
		return ones(s, DType::Bool);
	}else if(p == 0){
		return zeros(s, DType::Bool);
	}
	Tensor out = zeros(std::move(s), DType::Bool);
	Tensor range = arange(out.numel(), DType::int64, 0);
	int64_t n = out.numel();
	int64_t numOnes = static_cast<int64_t>(p * n);

	std::random_device rd;
	std::minstd_rand gen(rd());
	std::uniform_int_distribution<> dis(0, n - 1);
	
	int64_t* range_begin = reinterpret_cast<int64_t*>(range.data_ptr());
	int64_t* range_end = range_begin + n;
	std::shuffle(range_begin, range_end, gen);
	uint_bool_t* out_indices = reinterpret_cast<uint_bool_t*>(out.data_ptr());
	for (int64_t i = 0; i < numOnes; ++i) {
		out_indices[range_begin[i]] = uint_bool_t(true);
	}
	return std::move(out);
}

Tensor randn(SizeRef inp, DType dt){
	Tensor output = randint(0, 20, std::move(inp), dt);
	softmax_(output);
	return std::move(output);
}

}
}
