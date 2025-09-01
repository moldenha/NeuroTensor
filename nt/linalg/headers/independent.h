#ifndef NT_LINALG_INDEPENDENT_H__
#define NT_LINALG_INDEPENDENT_H__

#include "../../Tensor.h"
#include <Eigen/Dense>
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>
#include <string.h>

namespace nt {
namespace linalg {

//this is a function to find independent rows 
NEUROTENSOR_API Tensor indp_rows(const Tensor&, const Tensor&);
//this is a function to find independent columns 
NEUROTENSOR_API Tensor indp_cols(const Tensor&, const Tensor&);


} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_COLUMN_SPACE_H_
