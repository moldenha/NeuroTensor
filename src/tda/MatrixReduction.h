// these are functions to reduce a matrix specific to the format from boundary
// matrices
#ifndef _NT_TDA_MATRIX_REDUCTION_FUNCTIONAL_H_
#define _NT_TDA_MATRIX_REDUCTION_FUNCTIONAL_H_

#include "../Tensor.h"

namespace nt {
namespace tda {

Tensor simultaneousReduce(SparseTensor &d_k, SparseTensor &d_kplus1);
Tensor &finishRowReducing(Tensor &B);
int64_t numPivotRows(Tensor &A);
Tensor rowReduce(Tensor A);

} // namespace tda
} // namespace nt

#endif
