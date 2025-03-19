#include "headers/QR.h"
#include "headers/toEigen.hpp"
#include <type_traits>

namespace nt{
namespace linalg{

Tensor QR(Tensor _t){
    return runEigenFunction(_t, [](auto& mat) -> Tensor{
        using MatrixType = std::remove_cvref_t<decltype(mat)>;
        using ScalarType = typename MatrixType::Scalar;
        auto [Q, R] = QR_eigen<ScalarType>(mat);
        Tensor _Q = fromEigen(Q);
        Tensor _R = fromEigen(R);
        return functional::list(_Q, _R);
    });
}

}}
