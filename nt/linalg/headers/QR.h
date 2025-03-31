#ifndef _NT_LINALG_QR_H_
#define _NT_LINALG_QR_H_

#include "../../Tensor.h"
#include <functional>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>
#include <Eigen/Dense>
#include <tuple>

namespace nt {
namespace linalg {
Tensor QR(Tensor);
template<typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        QR_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&);

} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_QR_H_
