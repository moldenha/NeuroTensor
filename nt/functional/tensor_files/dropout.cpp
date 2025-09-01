#include "dropout.h"
#include "rand.h"
#include "exceptions.hpp"
#include "../cpu/dropout.h"

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
    utils::throw_exception(input.dims() > 2, "Error cannot perform a 2d dropout on a tensor with only $ dims", input.dims());
    Tensor out = input.clone();
    const int64_t& cols = out.shape()[-1];
    const int64_t& rows = out.shape()[-2];
    Tensor bools = randbools({out.numel() / (rows * cols)}, p);
    cpu::_dropout2d_(out.arr_void(), bools.arr_void(), rows, cols);
    return std::move(out);
}

Tensor dropout3d(const Tensor& input, double p){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    utils::throw_exception(input.dims() > 3, "Error cannot perform a 3d dropout on a tensor with only $ dims", input.dims());
    Tensor out = input.clone();
    const int64_t& cols = out.shape()[-1];
    const int64_t& rows = out.shape()[-2];
    const int64_t& channels = out.shape()[-3];
    Tensor bools = randbools({out.numel() / (channels * rows * cols)}, p);
    cpu::_dropout3d_(out.arr_void(), bools.arr_void(), channels, rows, cols);
    return std::move(out);
}


}
}
