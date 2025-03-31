#include "../../utils/utils.h"
#include "../headers/EigenDetails.hpp"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <type_traits>

namespace nt {
namespace linalg {

template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
QR_eigen(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mat) {
    using MatType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>;
    if constexpr (std::is_floating_point_v<T> || detail::is_eigen_complex<T>) {
        Eigen::HouseholderQR<MatType> qr(mat);
        MatType Q = qr.householderQ();
        MatType R = qr.matrixQR().template triangularView<Eigen::Upper>();
        return std::make_tuple(Q, R);

    } else {
        utils::throw_exception(false, "Types not complex or floating are not "
                                      "supported for QR Decomposition");
        MatType Q(0, 0);
        MatType R(0, 0);
        return std::make_tuple(Q, R);
    }
}

#define _NT_QR_EIGEN_DECLARE_(type)                                            \
    template std::tuple<                                                       \
        Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,  \
        Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  \
    QR_eigen<type>(Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,         \
                                 Eigen::RowMajor> &);
_NT_DECLARE_EIGEN_TYPES_(_NT_QR_EIGEN_DECLARE_);
#undef _NT_QR_EIGEN_DECLARE_

} // namespace linalg
} // namespace nt
