#ifndef _NT_SIMPLEX_RADI_FUNCTIONS_H_
#define _NT_SIMPLEX_RADI_FUNCTIONS_H_
#include "../Tensor.h"
#include "BasisOverlapping.h"

namespace nt {
namespace tda {

Tensor compute_circumradii(const Tensor &simplicies);
Tensor compute_circumradii(const Tensor &index_simplicies,
                           const Tensor &points);
Tensor compute_point_radii(Tensor simplicies);
Tensor compute_point_radii(const Tensor &index_simplicies,
                           const Tensor &points);
Tensor compute_point_radii(const Tensor &index_simplicies,
                           const BasisOverlapping &balls, int64_t batch);
void sort_simplex_on_radi(Tensor &simplicies, Tensor &simplex_radi);
std::set<double> get_radi_set(const Tensor &simplex_radi, int64_t batch = 0);

} // namespace tda
} // namespace nt

#endif
