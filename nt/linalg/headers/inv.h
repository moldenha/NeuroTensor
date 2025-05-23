#ifndef _NT_LINALG_INV_H_
#define _NT_LINALG_INV_H_

#include "../../Tensor.h"
#include "EigenDetails.hpp"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>
#include <Eigen/Dense>
#include <tuple>

namespace nt {
namespace linalg {
Tensor inv(Tensor);
//pseudo inverse
Tensor pinv(Tensor, Scalar tolerance = 1e-6);

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inv_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pinv_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&, T);

} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_INV_H_
