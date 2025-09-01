#ifndef NT_TDA_NN_PH_FILTRATION_H__
#define NT_TDA_NN_PH_FILTRATION_H__

#include "../../nn/TensorGrad.h"
#include <tuple>

namespace nt {
namespace tda {

// returns simplex complex, and a tensor grad associated with their radii
// this is for vector rips
// other simplex types are planned to be added in the near future
// k represents the number of simplexes to get
NEUROTENSOR_API std::tuple<Tensor, TensorGrad> VRfiltration(const TensorGrad &dist_matrix, int64_t k, double max_radi = -1.0, bool sort=true);
//this is for a vector metric space where there are multiple distances outputted from a metric
NEUROTENSOR_API std::tuple<Tensor, TensorGrad, TensorGrad> VRfiltration(const TensorGrad& dist_matrix1, const TensorGrad& dist_matrix2, int64_t k, double max_radi = -1.0, bool sort = true);


} // namespace tda
} // namespace nt

#endif
