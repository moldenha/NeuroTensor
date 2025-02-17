#ifndef _NT_LINALG_H_
#define _NT_LINALG_H_

#include "../Tensor.h"


#include <vector>
#include <functional>
#include <optional>
#include "functional_matmult.h"
#include "functional_fold.h"
#include "functional_conv.h"
#include "../utils/optional_list.h"
#include <tuple>
#include <variant>

namespace nt{
namespace linalg{

/* std::tuple<Tensor, Tensor> qr_decomposition(const Tensor&); */
Tensor eye(int64_t n, int64_t b=0, DType dtype = DType::Float32); //make the identity matrix, b means batches
Tensor eye_like(const Tensor&);
Tensor norm(const Tensor& A, std::variant<std::nullptr_t, std::string, int64_t> ord = nullptr, utils::optional_list dim = nullptr, bool keepdim = false);
Tensor determinant(const Tensor&);
Tensor adjugate(const Tensor&);

}} //nt::linalg::


#endif //_NT_LINALG_H_
