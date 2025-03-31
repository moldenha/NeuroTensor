// this is a header file for constructing boundary matrices in terms of sparse
// tensors from simplices of dimension k and k+1
#ifndef _NT_TDA_BOUNDARIES_FUNCTIONAL_H_
#define _NT_TDA_BOUNDARIES_FUNCTIONAL_H_

#include "../sparse/SparseTensor.h"
#include "../Tensor.h"

namespace nt {
namespace tda {

// this works with indexes of simplex complexes
// it is faster, and if can be used should
SparseTensor compute_boundary_matrix_index(const Tensor &s_kp1,
                                               const Tensor &s_k);
// this can work with simplex complexes of ND points
SparseTensor compute_boundary_matrix(const Tensor &s_kp1,
                                         const Tensor &s_k);

} // namespace tda
} // namespace nt

#endif
