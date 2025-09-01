#ifndef NT_FUNCTIONAL_H__
#define NT_FUNCTIONAL_H__
#include "../Tensor.h"

#include "../dtype/DType.h"
#include "../utils/optional_list.h"
#include "tensor_files/activation_functions.h"
#include "tensor_files/colim_transform.h"
#include "tensor_files/combinations.h"
#include "tensor_files/combine.h"
#include "tensor_files/compare.h"
#include "tensor_files/complex.h"
#include "tensor_files/conv.h"
#include "tensor_files/convert.h"
#include "tensor_files/dilate.h"
#include "tensor_files/dropout.h"
#include "tensor_files/fill.h"
#include "tensor_files/flip.h"
#include "tensor_files/fused.h"
#include "tensor_files/index.h"
#include "tensor_files/matmult.h"
#include "tensor_files/mesh.h"
#include "tensor_files/min_max.h"
#include "tensor_files/normalize.h"
#include "tensor_files/numpy.h"
#include "tensor_files/operators.h"
#include "tensor_files/padding.h"
#include "tensor_files/pooling.h"
#include "tensor_files/print.h"
#include "tensor_files/rand.h"
#include "tensor_files/ranges.h"
#include "tensor_files/repeat.h"
#include "tensor_files/round.h"
#include "tensor_files/save_load.h"
#include "tensor_files/softmax.h"
#include "tensor_files/sort.h"
#include "tensor_files/split.h"
#include "tensor_files/stride.h"
#include "tensor_files/sum_exp_log.h"
#include "tensor_files/transpose.h"
#include "tensor_files/trig.h"
#include "tensor_files/unique.h"
#include <cstring>
#include <functional>
#include <optional>
#include <vector>

namespace std {
inline ::nt::Tensor cos(const ::nt::Tensor &t) {
    return ::nt::functional::cos(t);
}
inline ::nt::Tensor sin(const ::nt::Tensor &t) {
    return ::nt::functional::sin(t);
}
inline ::nt::Tensor tan(const ::nt::Tensor &t) {
    return ::nt::functional::tan(t);
}
inline ::nt::Tensor cosh(const ::nt::Tensor &t) {
    return ::nt::functional::cosh(t);
}
inline ::nt::Tensor sinh(const ::nt::Tensor &t) {
    return ::nt::functional::sinh(t);
}
inline ::nt::Tensor tanh(const ::nt::Tensor &t) {
    return ::nt::functional::tanh(t);
}
inline ::nt::Tensor sqrt(const ::nt::Tensor &t) {
    return ::nt::functional::sqrt(t);
}
inline ::nt::Tensor abs(const ::nt::Tensor &t) {
    return ::nt::functional::abs(t);
}
inline ::nt::Tensor log(const ::nt::Tensor &t) {
    return ::nt::functional::log(t);
}
} // namespace std

#endif
