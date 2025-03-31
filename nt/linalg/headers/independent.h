#ifndef _NT_LINALG_INDEPENDENT_H_
#define _NT_LINALG_INDEPENDENT_H_

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
Tensor indp_rows(const Tensor&, const Tensor&);
//this is a function to find independent columns 
Tensor indp_cols(const Tensor&, const Tensor&);


} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_COLUMN_SPACE_H_
