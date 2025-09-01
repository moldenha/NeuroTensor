#ifndef NT_LINALG_SVD_H__
#define NT_LINALG_SVD_H__

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
NEUROTENSOR_API Tensor SVD(Tensor);
template<typename T>
std::tuple<Eigen::Matrix<detail::from_complex_eigen_t<T>, Eigen::Dynamic, 1>,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
            SVD_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);

} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_SVD_H_
