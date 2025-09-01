#include "headers/inv.h"
#include "headers/toEigen.hpp"
#include "../functional/functional.h"
#include <type_traits>

namespace nt{
namespace linalg{

Tensor inv(Tensor _t){
    utils::throw_exception(_t.dims() == 2, "Expected to get matrix when taking the inverse");
    utils::throw_exception(_t.shape()[0] == _t.shape()[1], "Can only take the inverse of square matricies, but got shape $", _t.shape());
    return runEigenFunction(_t, [](auto& mat) -> Tensor{
        using MatrixType = ::nt::type_traits::remove_cvref_t<decltype(mat)>;
        using ScalarType = typename MatrixType::Scalar;
        auto out = inv_eigen<ScalarType>(mat);
        return fromEigen(out);
    });
}

Tensor pinv(Tensor _t, Scalar tolerance){
    return runEigenFunction(_t, [&tolerance](auto& mat) -> Tensor{
        using MatrixType = ::nt::type_traits::remove_cvref_t<decltype(mat)>;
        using ScalarType = typename MatrixType::Scalar;
        if constexpr (std::is_same_v<ScalarType, std::complex<float> >){
            complex_64 c = tolerance.to<complex_64>();
            auto out = pinv_eigen<ScalarType>(mat, std::complex<float>(c));
            return fromEigen(out);
        }
        else if constexpr (std::is_same_v<ScalarType, std::complex<double> >){
            complex_128 c = tolerance.to<complex_128>();
            auto out = pinv_eigen<ScalarType>(mat, std::complex<double>(c));
            return fromEigen(out);
        }
        else{
            auto out = pinv_eigen<ScalarType>(mat, tolerance.to<ScalarType>());
            return fromEigen(out);
        }
    });
}

}}
