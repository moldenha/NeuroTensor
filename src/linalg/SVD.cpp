#include "headers/SVD.h"
#include "headers/toEigen.hpp"
#include "../functional/functional.h"
#include <type_traits>

namespace nt{
namespace linalg{

Tensor SVD(Tensor _t){
    return runEigenFunction(_t, [](auto& mat) -> Tensor{
        using MatrixType = std::remove_cvref_t<decltype(mat)>;
        using ScalarType = typename MatrixType::Scalar;
        auto [S, U, V] = SVD_eigen<ScalarType>(mat);
        Tensor _S = fromEigen(S);
        Tensor _U = fromEigen(U);
        Tensor _V = fromEigen(V);
        return functional::list(_S, _U, _V);
    });
}

}}
