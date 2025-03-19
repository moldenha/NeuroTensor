#include <iostream>
#include <Eigen/Dense>
#include <complex>
#include "../headers/EigenDetails.hpp"
#include <type_traits>
#include <tuple>

namespace nt{
namespace linalg{

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    svd_col_space_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat){
    using MatType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    if constexpr (detail::is_eigen_complex<T> || std::is_floating_point_v<T>){
        Eigen::JacobiSVD<MatType> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // Extract column space (basis of range(A))
        return svd.matrixU();
    }
    else{
        utils::throw_exception(false, "Can only get null space with SVD from floating types");
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V(0,0);
        return V;
    }
}


template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    lu_col_space_eigen(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat){
    using MatType = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    if constexpr (detail::is_eigen_complex<T> || std::is_floating_point_v<T>){
        Eigen::FullPivLU<MatType> lu(mat);
        return mat * lu.permutationP();
    }
    else{
        utils::throw_exception(false, "Can only get null space with LU Decomposition from floating types");
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V(0,0);
        return V;
    }
}


#define _NT_COLUMN_SPACE_EIGEN_DECLARE_(type) template Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> \
        svd_col_space_eigen<type>(Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&); \
        template Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> \
        lu_col_space_eigen<type>(Eigen::Matrix<type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&); \

_NT_DECLARE_EIGEN_TYPES_(_NT_COLUMN_SPACE_EIGEN_DECLARE_);
#undef _NT_COLUMN_SPACE_EIGEN_DECLARE_

}}
