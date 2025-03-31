#ifndef __NT_TDA_NN_PH_FILTRATION_H__
#define __NT_TDA_NN_PH_FILTRATION_H__

#include "../../nn/TensorGrad.h"
#include <tuple>

namespace nt {
namespace tda {

// returns simplex complex, and a tensor grad associated with their radii
// this is for vector rips
// other simplex types are planned to be added in the near future
// k represents the number of simplexes to get
std::tuple<Tensor, TensorGrad> VRfiltration(const TensorGrad &dist_matrix, int64_t k);

} // namespace tda
} // namespace nt

#endif
