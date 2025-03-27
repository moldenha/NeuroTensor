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
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
svd_null_space_eigen(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mat) {
    using MatType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>;
    if constexpr (detail::is_eigen_complex<T>) {
        Eigen::JacobiSVD<MatType> svd(mat, Eigen::ComputeFullV);
        Eigen::Matrix<detail::from_complex_eigen_t<T>, Eigen::Dynamic, 1>
            singularValues =
                svd.singularValues().real(); // Singular values are always real
        // Tolerance for zero singular values
        detail::from_complex_eigen_t<T> tol = 1e-10;
        // Determine the null space basis (right singular vectors corresponding
        // to zero singular values)
        int rank = (singularValues.array() > tol).count();
        int nullity = mat.cols() - rank;

        return svd.matrixV().rightCols(nullity);
    } else if constexpr (std::is_floating_point_v<T>) {
        Eigen::JacobiSVD<MatType> svd(mat, Eigen::ComputeFullV);
        Eigen::Matrix<T, Eigen::Dynamic, 1> singularValues =
            svd.singularValues(); // Singular values are always real
        // Tolerance for zero singular values
        T tol = 1e-10;
        // Determine the null space basis (right singular vectors corresponding
        // to zero singular values)
        int rank = (singularValues.array() > tol).count();
        int nullity = mat.cols() - rank;
        return svd.matrixV().rightCols(nullity);
    } else {
        utils::throw_exception(
            false, "Can only get null space with SVD from floating types");
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V(0,
                                                                            0);
        return V;
    }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
lu_null_space_eigen(
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mat) {
    using MatType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>;
    if constexpr (detail::is_eigen_complex<T> || std::is_floating_point_v<T>) {
        Eigen::FullPivLU<MatType> lu(mat);
        return lu.kernel(); // Eigen's built-in null space computation
    } else {
        utils::throw_exception(false, "Can only get null space with LU "
                                      "Decomposition from floating types");
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V(0,
                                                                            0);
        return V;
    }
}

#define _NT_NULL_SPACE_EIGEN_DECLARE_(type)                                    \
    template Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,               \
                           Eigen::RowMajor>                                    \
    svd_null_space_eigen<type>(                                                \
        Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>   \
            &);                                                                \
    template Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic,               \
                           Eigen::RowMajor>                                    \
    lu_null_space_eigen<type>(                                                 \
        Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>   \
            &);

_NT_DECLARE_EIGEN_TYPES_(_NT_NULL_SPACE_EIGEN_DECLARE_);
#undef _NT_NULL_SPACE_EIGEN_DECLARE_

} // namespace linalg
} // namespace nt
