#ifndef _NT_LINALG_SVD_H_
#define _NT_LINALG_SVD_H_

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
Tensor SVD(Tensor);
template<typename T>
std::tuple<Eigen::Matrix<detail::from_complex_eigen_t<T>, Eigen::Dynamic, 1>,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
            SVD_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);

} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_SVD_H_
