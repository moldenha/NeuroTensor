#ifndef NT_LINALG_PROCESS_SVD_HPP__
#define NT_LINALG_PROCESS_SVD_HPP__ 

#include "../../utils/utils.h"
#include "../headers/EigenDetails.hpp"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace nt {
namespace linalg {

template <typename T>
std::tuple<Eigen::Matrix<detail::from_complex_eigen_t<T>, Eigen::Dynamic, 1>,
           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
SVD_eigen(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mat) {
    using MatType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>;
    if constexpr (std::is_same_v<std::complex<double>, T>) {
        Eigen::JacobiSVD<MatType> svd(mat, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
        Eigen::VectorXd singularValues =
            svd.singularValues().real(); // Only real part
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
            U = svd.matrixU();
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
            V = svd.matrixV();
        return std::make_tuple(singularValues, U, V);
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
        Eigen::JacobiSVD<MatType> svd(mat, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
        Eigen::Matrix<float, Eigen::Dynamic, 1> singularValues =
            svd.singularValues().real(); // Only real part
        Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
            U = svd.matrixU();
        Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
            V = svd.matrixV();
        return std::make_tuple(singularValues, U, V);
    } else if constexpr (std::is_same_v<float, T> ||
                         std::is_same_v<double, T>) {
        Eigen::JacobiSVD<MatType> svd(mat, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
        Eigen::Matrix<T, Eigen::Dynamic, 1> singularValues =
            svd.singularValues(); // Only real part
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U =
            svd.matrixU();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V =
            svd.matrixV();
        return std::make_tuple(singularValues, U, V);
    } else {
        utils::throw_exception(false, "SVD only works with floatin types");
        Eigen::Matrix<T, Eigen::Dynamic, 1> singularValues(0); // Only real part
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U(0,
                                                                            0);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V(0,
                                                                            0);
        return std::make_tuple(singularValues, U, V);
    }
}

#define _NT_SVD_EIGEN_DECLARE_(type)                                           \
    template std::tuple<                                                       \
        Eigen::Matrix<detail::from_complex_eigen_t<type>, Eigen::Dynamic, 1>,  \
        Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,  \
        Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  \
    SVD_eigen<type>(Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,        \
                                  Eigen::RowMajor> &);

// _NT_DECLARE_EIGEN_TYPES_(_NT_SVD_EIGEN_DECLARE_);
// #undef _NT_SVD_EIGEN_DECLARE_

} // namespace linalg
} // namespace nt

#endif
