#ifndef NT_LINALG_COLUMN_SPACE_H__
#define NT_LINALG_COLUMN_SPACE_H__

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


//can find the column space from SVD decomposition or LU decomposition
//  Accepted: "svd" or "lu"
NEUROTENSOR_API Tensor col_space(const Tensor&, std::string mode = "lu");

//this takes a reduced matrix, finds the pivot columns from the reduced space, and then gets the column space from original
NEUROTENSOR_API Tensor col_space(const Tensor& original, const Tensor& reduced_space);

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    lu_col_space_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    svd_col_space_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);

} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_COLUMN_SPACE_H_
