#ifndef _NT_TDA_SIMPLEX_CONSTRUCT_FUNCTIONAL_H_
#define _NT_TDA_SIMPLEX_CONSTRUCT_FUNCTIONAL_H_

#include "../Tensor.h"
#include "BasisOverlapping.h"

namespace nt {
namespace tda {

Tensor find_all_simplicies_indexes(int64_t simplicies_amt, const Tensor &points,
                                   const BasisOverlapping &balls,
                                   double radius);
Tensor from_index_simplex_to_point_simplex(const Tensor &indexes,
                                           const Tensor &points);
inline Tensor find_all_simplicies(int64_t simplicies_amt, const Tensor &points,
                                  const BasisOverlapping &balls, double radius,
                                  bool indexes_only = false) {
    Tensor indexes =
        find_all_simplicies_indexes(simplicies_amt, points, balls, radius);
    if (indexes_only) {
        return std::move(indexes);
    }
    return from_index_simplex_to_point_simplex(indexes, points);
}

Tensor find_all_simplicies(int64_t simplicies_amt, const Tensor &points,
                           const BasisOverlapping &balls,
                           bool indexes_only = false);

std::pair<Tensor, Tensor> find_all_simplicies(int64_t simplicies_amt, const int64_t num_points,
                           const Tensor &distance_matrix, double max_radi=-1.0, bool sort = true);
} // namespace tda
} // namespace nt

#endif
