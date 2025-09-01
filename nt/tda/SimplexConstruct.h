#ifndef NT_TDA_SIMPLEX_CONSTRUCT_FUNCTIONAL_H__
#define NT_TDA_SIMPLEX_CONSTRUCT_FUNCTIONAL_H__

#include "../Tensor.h"
#include "BasisOverlapping.h"

namespace nt {
namespace tda {

NEUROTENSOR_API Tensor find_all_simplicies_indexes(int64_t simplicies_amt, const Tensor &points,
                                   const BasisOverlapping &balls,
                                   double radius);
NEUROTENSOR_API Tensor from_index_simplex_to_point_simplex(const Tensor &indexes,
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

NEUROTENSOR_API Tensor find_all_simplicies(int64_t simplicies_amt, const Tensor &points,
                           const BasisOverlapping &balls,
                           bool indexes_only = false);

NEUROTENSOR_API std::pair<Tensor, Tensor> find_all_simplicies(int64_t simplicies_amt, const int64_t num_points,
                           const Tensor &distance_matrix, double max_radi=-1.0, bool sort = true);
} // namespace tda
} // namespace nt

#endif
