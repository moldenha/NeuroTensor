#ifndef NT_LINALG_NULL_SPACE_H__
#define NT_LINALG_NULL_SPACE_H__

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
//an optimized route for taking the null space of an already reduced matrix
//reduced_null_space(t, true) is the same as reduced_null_space(t.transpose(-1, -2), false).transpose(-1, -2)
//pivot rows is to look for pivot rows instead of columns
//reduced_null_space(t, true, true) is the same as reduced_null_space(t.transpose(-1, -2), false, false) 
NEUROTENSOR_API Tensor reduced_null_space(const Tensor&, bool pivot_rows=false, bool pivots_first = false);

//can find the null space from SVD decomposition or LU decomposition
//  Accepted: "svd" or "lu"
NEUROTENSOR_API Tensor null_space(const Tensor&, std::string mode = "svd");

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    lu_null_space_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    svd_null_space_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);

} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_NULL_SPACE_H_
