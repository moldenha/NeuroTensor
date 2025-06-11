#ifndef _NT_LINALG_PIVOTS_H_
#define _NT_LINALG_PIVOTS_H_

#include "../../Tensor.h"

namespace nt {
namespace linalg {
Tensor pivot_rows(const Tensor&);
Tensor pivot_cols(const Tensor&, bool return_where=false);

Tensor extract_free_cols(const Tensor&);
Tensor extract_pivot_cols(const Tensor&);
Tensor extract_free_rows(const Tensor&);
Tensor extract_pivot_rows(const Tensor&);

//returns the number of pivot cols or rows in every matrix
Tensor num_pivot_rows(const Tensor&);
Tensor num_pivot_cols(const Tensor&);

//this is able to just return a number by only taking a matrix
int64_t num_pivot_rows_matrix(const Tensor&);
int64_t num_pivot_cols_matrix(const Tensor&);
} // namespace linalg
} // namespace nt

#endif //_NT_LINALG_PIVOTS_H_
