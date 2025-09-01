#ifndef NT_SIMPLEX_RADI_FUNCTIONS_H__
#define NT_SIMPLEX_RADI_FUNCTIONS_H__
#include "../Tensor.h"
#include "BasisOverlapping.h"

namespace nt {
namespace tda {

NEUROTENSOR_API Tensor compute_circumradii(const Tensor &simplicies);
NEUROTENSOR_API Tensor compute_circumradii(const Tensor &index_simplicies,
                           const Tensor &points);
NEUROTENSOR_API Tensor compute_point_radii(Tensor simplicies);
NEUROTENSOR_API Tensor compute_point_radii(const Tensor &index_simplicies,
                           const Tensor &points);
NEUROTENSOR_API Tensor compute_point_radii(const Tensor &index_simplicies,
                           const BasisOverlapping &balls, int64_t batch);
NEUROTENSOR_API std::pair<Tensor, Tensor> compute_point_grad_radii(const Tensor& index_simplicies,
                                               const Tensor& distances);
NEUROTENSOR_API void sort_simplex_on_radi(Tensor &simplicies, Tensor &simplex_radi, Tensor& grad_indexes);
NEUROTENSOR_API void sort_simplex_on_radi(Tensor &simplicies, Tensor &simplex_radi);
NEUROTENSOR_API std::set<double> get_radi_set(const Tensor &simplex_radi, int64_t batch = 0);



} // namespace tda
} // namespace nt

#endif
