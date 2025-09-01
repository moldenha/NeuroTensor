#ifndef NT_LINALG_PIVOTS_H__
#define NT_LINALG_PIVOTS_H__

#include "../../Tensor.h"

namespace nt {
namespace linalg {
NEUROTENSOR_API Tensor pivot_rows(const Tensor&);
NEUROTENSOR_API Tensor pivot_cols(const Tensor&, bool return_where=false);

NEUROTENSOR_API Tensor extract_free_cols(const Tensor&);
NEUROTENSOR_API Tensor extract_pivot_cols(const Tensor&);
NEUROTENSOR_API Tensor extract_free_rows(const Tensor&);
NEUROTENSOR_API Tensor extract_pivot_rows(const Tensor&);

//returns the number of pivot cols or rows in every matrix
NEUROTENSOR_API Tensor num_pivot_rows(const Tensor&);
NEUROTENSOR_API Tensor num_pivot_cols(const Tensor&);

//this is able to just return a number by only taking a matrix
NEUROTENSOR_API int64_t num_pivot_rows_matrix(const Tensor&);
NEUROTENSOR_API int64_t num_pivot_cols_matrix(const Tensor&);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_PIVOTS_H_
