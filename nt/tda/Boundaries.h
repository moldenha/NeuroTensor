// this is a header file for constructing boundary matrices in terms of sparse
// tensors from simplices of dimension k and k+1
#ifndef NT_TDA_BOUNDARIES_FUNCTIONAL_H__
#define NT_TDA_BOUNDARIES_FUNCTIONAL_H__

#include "../sparse/SparseTensor.h"
#include "../sparse/SparseMatrix.h"
#include "../Tensor.h"
#include <tuple>

namespace nt {
namespace tda {

// this works with indexes of simplex complexes
// it is faster, and if can be used should
NEUROTENSOR_API SparseTensor compute_boundary_matrix_index(const Tensor &s_kp1,
                                               const Tensor &s_k);
// this can work with simplex complexes of ND points
NEUROTENSOR_API SparseTensor compute_boundary_matrix(const Tensor &s_kp1,
                                         const Tensor &s_k);
NEUROTENSOR_API SparseMatrix compute_boundary_sparse_matrix_index(const Tensor& s_kp1,
                                            const Tensor& s_k);

NEUROTENSOR_API std::tuple<
    std::vector<int64_t>, //x indexes
    std::vector<int64_t>, //y indexes
    std::vector<float>    //boundaries
        > compute_differentiable_boundary_sparse_matrix_index(const Tensor &s_kp1,
                                           const Tensor &s_k, 
                                            const Tensor& rkp1, const Tensor& rk); 

} // namespace tda
} // namespace nt

#endif
