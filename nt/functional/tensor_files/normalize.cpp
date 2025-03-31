#include "normalize.h"
#include "../cpu/normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"
#include <cmath>

namespace nt{
namespace functional{

void xavier_uniform_(Tensor& tensor){
    utils::throw_exception(tensor.dims() >= 2, "For xavier uniform the dimensions of the tensor must be greater than or equal to 2");
    int64_t fan_in = tensor.shape()[-1]; //switch to [1] maybe
    int64_t fan_out = tensor.shape()[-2]; //switch to [0] maybe
    double bound = std::sqrt(6.0 / (double)(fan_in + fan_out));
    cpu::xavier_uniform_(tensor.arr_void(), bound);
}

}
}
