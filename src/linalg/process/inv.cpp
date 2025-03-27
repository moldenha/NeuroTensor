#include "../../utils/utils.h"
#include "../headers/EigenDetails.hpp"
#include "../headers/SVD.h"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace nt {
namespace linalg {

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inv_eigen(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mat) {
    if constexpr (detail::is_eigen_complex<T> || std::is_floating_point_v<T>) {
        if (mat.rows() == mat.cols()) {
            return mat.fullPivLu().inverse();
        } else {
            return mat.llt().solve(
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>::Identity(mat.rows(),
                                                         mat.cols()));
        }
    } else {
        utils::throw_exception(false, "pinv only works with floatin types");
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U(0,
                                                                            0);
        return std::move(U);
    }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pinv_eigen(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mat,
    T tolerance) {
    using MatrixType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                              Eigen::RowMajor>;
    if constexpr (detail::is_eigen_complex<T> || std::is_floating_point_v<T>) {
        return mat.completeOrthogonalDecomposition().pseudoInverse();
        // Eigen::JacobiSVD<MatrixType> svd(mat, Eigen::ComputeThinU |
        // Eigen::ComputeThinV); auto singularValues = svd.singularValues();
        // MatrixType S_inv = MatrixType::Zero(mat.cols(), mat.rows()); //
        // Transposed dimensions
        // // Invert non-zero singular values
        // detail::from_complex_eigen_t<T> tl;
        // if constexpr (detail::is_eigen_complex<T>){
        //     tl = tolerance.real();
        // }else{
        //     tl = tolerance;
        // }
        // for (int i = 0; i < singularValues.size(); ++i) {
        //     if (singularValues(i) > tl) {
        //         S_inv(i, i) = 1.0 / singularValues(i);
        //     }
        // }

        // return svd.matrixV() * S_inv * svd.matrixU().transpose();
    } else {
        utils::throw_exception(false, "pinv only works with floatin types");
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U(0,
                                                                            0);
        return std::move(U);
    }
}

#define _NT_INV_EIGEN_DECLARE_(type)                                           \
    template Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,               \
                           Eigen::RowMajor>                                    \
    inv_eigen<type>(Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,        \
                                  Eigen::RowMajor> &);                         \
    template Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,               \
                           Eigen::RowMajor>                                    \
    pinv_eigen<type>(Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,       \
                                   Eigen::RowMajor> &,                         \
                     type);

_NT_DECLARE_EIGEN_TYPES_(_NT_INV_EIGEN_DECLARE_);
#undef _NT_INV_EIGEN_DECLARE_

} // namespace linalg
} // namespace nt
