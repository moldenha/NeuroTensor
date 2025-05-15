#ifndef __NT_TDA_NN_LAPLACIAN_H__
#define __NT_TDA_NN_LAPLACIAN_H__
#include "../../nn/TensorGrad.h"

// this is a header file designed to find the hodge laplacian and make it
// differentiable it can be used to learn paths

namespace nt {
namespace tda {

// this takes k-1 radi, k radi, and k+1 radi and returns a laplacian
TensorGrad hodge_laplacian(TensorGrad radi_1, TensorGrad radi_2,
                           TensorGrad radi_3, Tensor simplex_complex_1,
                           Tensor simplex_complex_2, Tensor simplex_complex_3);
// this takes a distance matrix, and returns a hodge laplacian differentiable
std::tuple<TensorGrad, Tensor> hodge_laplacian(TensorGrad distance_matrix, int64_t k, double max_radi=-1.0);
std::tuple<TensorGrad, Tensor> hodge_laplacian(TensorGrad distance_matrix, TensorGrad distance_matrix2, int64_t k, double max_radi=-1.0);

TensorGrad findAllPaths(const TensorGrad& laplacian);

} // namespace tda
} // namespace nt

#endif
