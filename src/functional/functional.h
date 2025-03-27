#ifndef _NT_FUNCTIONAL_H_
#define _NT_FUNCTIONAL_H_
#include "../Tensor.h"

#include "../dtype/DType.h"
#include "../utils/optional_list.h"
#include "tensor_files/colim_transform.h"
#include "tensor_files/combine.h"
#include "tensor_files/conv.h"
#include "tensor_files/fill.h"
#include "tensor_files/fused.h"
#include "tensor_files/save_load.h"
#include "tensor_files/matmult.h"
#include "tensor_files/normalize.h"
#include "tensor_files/numpy.h"
#include "tensor_files/operators.h"
#include "tensor_files/rand.h"
#include <cstring>
#include <functional>
#include <optional>
#include <vector>


namespace nt {
namespace functional {

Tensor dot(const Tensor &, const Tensor &, utils::optional_list dim = nullptr,
           bool keepdim = false);

bool all(const Tensor &);
bool any(const Tensor &);
Tensor all(const Tensor, int64_t dim);
Tensor any(const Tensor, int64_t dim);
void save(const Tensor &, const char *);
Tensor load(const char *);

Tensor conv2dT(const Tensor &image, const Tensor &kernel,
               const uint32_t s_h = 1, const uint32_t s_w = 1);
void softmax_(Tensor &);
void softmax_(Tensor &, uint32_t);
void softmax_stable_(Tensor &);
void softmax_stable_(Tensor &, uint32_t);
Tensor dropout(const Tensor &, double);
Tensor sigmoid(const Tensor &);
Tensor dsigmoid(const Tensor &, bool apply_sigmoid = true);
Tensor silu(const Tensor &);
Tensor dsilu(const Tensor &);
Tensor gelu(const Tensor &);
Tensor dgelu(const Tensor &);
Tensor tan(const Tensor &);
Tensor tanh(const Tensor &);
Tensor sin(const Tensor &);
Tensor sinh(const Tensor &);
Tensor cos(const Tensor &);
Tensor cosh(const Tensor &);
Tensor dtan(const Tensor &);  // derivative of tan
Tensor dtanh(const Tensor &); // derivative of tanh
Tensor sqrt(const Tensor &);
Tensor invsqrt(const Tensor &);  // 1 / sqrt(x);
Tensor dinvsqrt(const Tensor &); // derivative of invsqrt
Tensor var(const Tensor &, utils::optional_list dim = nullptr,
           int64_t correction = 1,
           bool keepdim = false); // delta degrees of freedom (0 for population
                                  // variance, 1 for sample variance).
Tensor dvar(const Tensor &dx, const Tensor &x,
            utils::optional_list dim = nullptr,
            int64_t correction = 1); // derivative of the var function with
                                     // respect to xi element of the the tensor
Tensor abs(const Tensor &); // absolute value
Tensor log(const Tensor &);
Tensor dlog(const Tensor &);
Tensor clamp(const Tensor &x, std::optional<int64_t> min = std::nullopt,
             std::optional<int64_t> max = std::nullopt);
Tensor relu(const Tensor &);
Tensor softplus(const Tensor &x, Scalar beta = 1.0, Scalar threshold = 20.0);
Tensor softmax(Tensor &);
Tensor softmax(Tensor &, uint32_t);
Tensor softmax_stable(Tensor &);
Tensor softmax_stable(Tensor &, uint32_t);

size_t amount_of(const Tensor &, Scalar);
size_t count(const Tensor &);
Tensor where(Tensor);
Tensor index_select(Tensor input, int8_t dim, Tensor index);
Tensor select(Tensor input, int8_t dim, typename Tensor::size_value_t index);
Tensor meshgrid(Tensor &&, Tensor &&);
Tensor split(Tensor input, typename Tensor::size_value_t split_size,
             int64_t dim = 0); // splits into variable number of split sizes
                               // along a given dimension
Tensor
split(Tensor input, std::vector<typename Tensor::size_value_t> split_sections,
      int64_t dim = 0); // splits into a specified amount on the given dimension
Tensor chunk(Tensor input, typename Tensor::size_value_t chunks,
             int64_t dim = 0); // splits into that many chunks
Tensor as_strided(const Tensor &input, const SizeRef n_size, SizeRef n_stride,
                  const int64_t storage_offset = 0, bool whole_tensor = false);
Tensor sort(const Tensor &, const Tensor::size_value_t dim = -1,
            bool descending = false, bool return_sorted = true,
            bool return_indices = true);
// returns Tensor(sorted, indices)
// only returning the indices or the sorted makes things faster
// this function is meant to sort along for example each row in a matrix or each
// collumn
Tensor coordsort(const Tensor &input, Tensor::size_value_t dim = -2,
                 bool descending = false, bool return_sorted = true,
                 bool return_indices = true);
Tensor unique(Tensor, int64_t dim, bool return_unique = true,
              bool return_indices = true);
// returns Tensor(unique, indices) (depending on return_unique and
// return_indices)
// puts the tensors into a Tensor of dtype TensorObj with sizeof...(Args) + 1
// number of tensors
Tensor combinations(Tensor vec, int64_t r, int64_t start = 0);

} // namespace functional
} // namespace nt

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
