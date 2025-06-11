#include "dropout.h"
#include "rand.h"
#include "exceptions.hpp"

namespace nt{
namespace functional{

Tensor dropout(const Tensor& input, double p){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
	Tensor bools = randbools(input.shape(), p);
	Tensor out = input.clone();
	out[bools] = 0;
	return std::move(out);
}

Tensor dropout2d(const Tensor& input, double p){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    Tensor out = input.clone();
    Tensor split = out.split_axis(-2);
    Tensor bools = randbools(split.shape(), p);
    split[bools] = 0;
    return std::move(out);
}

Tensor dropout3d(const Tensor& input, double p){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    Tensor out = input.clone();
    Tensor split = out.split_axis(-2);
    Tensor bools = randbools(split.shape(), p);
    split[bools] = 0;
    return std::move(out);
}


}
}
