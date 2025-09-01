// these are functions to reduce a matrix specific to the format from boundary
// matrices
#ifndef NT_TDA_MATRIX_REDUCTION_FUNCTIONAL_H__
#define NT_TDA_MATRIX_REDUCTION_FUNCTIONAL_H__
#include "../Tensor.h"
#include "../functional/functional.h"
#include "../sparse/SparseTensor.h"
#include <map>
#include <tuple>

namespace nt {
namespace tda {

NEUROTENSOR_API Tensor simultaneousReduce(SparseTensor &d_k, SparseTensor &d_kplus1);
NEUROTENSOR_API Tensor &finishRowReducing(Tensor &B);
NEUROTENSOR_API int64_t numPivotRows(Tensor &A);
NEUROTENSOR_API Tensor rowReduce(Tensor A);

//these are functions to partially reduce boundary matrices
NEUROTENSOR_API Tensor simultaneousCatReduce(Tensor &d_k, Tensor &d_kplus1, int64_t start_rows,
                             int64_t start_cols, int64_t end_rows,
                             int64_t end_cols);
NEUROTENSOR_API Tensor &finishCatRowReducing(Tensor &B, int64_t start_rows, int64_t start_cols,
                             int64_t end_rows, int64_t end_cols);

NEUROTENSOR_API Tensor &partialColReduce(Tensor &d_k, int64_t start_rows, int64_t start_cols,
                         int64_t end_rows, int64_t end_cols);
NEUROTENSOR_API Tensor &partialRowReduce(Tensor &B, int64_t start_rows, int64_t start_cols,
                         int64_t end_rows, int64_t end_cols);

// this takes 2 boundary matrices, a map of radii corresponding to bounds of the
// boundary matrices that make up radi bounds, and then returns betti numbers
// corresponding to each radius
NEUROTENSOR_API std::map<double, int64_t> getBettiNumbers(
    SparseTensor &d_k, SparseTensor &d_kplus1,
    std::map<double, std::tuple<int64_t, int64_t, int64_t>> radi_bounds,
    double max = -1.0, bool add_zeros = false);

// when computing generators from boundary matrices, the column space of
// d_kplus1 needs to be created this uses the row-reduced d_kplus1 to make a
// column space for d_kplus1 at a specific radius where the amount of betti
// numbers is > 0
NEUROTENSOR_API std::pair<std::map<double, int64_t>, std::map<double, Tensor>> getBettiNumbersColSpace(
    SparseTensor &d_k, SparseTensor &d_kplus1,
    std::map<double, std::tuple<int64_t, int64_t, int64_t>> radi_bounds,
    double max = -1.0, bool add_zeros=false);

} // namespace tda
} // namespace nt

#endif
