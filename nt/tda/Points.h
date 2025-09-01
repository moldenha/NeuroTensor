#ifndef NT_TDA_POINTS_H__
#define NT_TDA_POINTS_H__

#include "../Tensor.h"
#include <cstdlib>

namespace nt {
namespace tda {

inline Tensor generate_random_cloud(SizeRef shape, double percent = 0.03){
    Tensor cloud = functional::zeros(std::move(shape), DType::int8);
    Tensor bools = functional::randbools(cloud.shape(), percent);
    cloud[bools] = 1;
    return std::move(cloud);
}
NEUROTENSOR_API Tensor extract_points_from_cloud(Tensor cloud, Scalar point, int64_t dims);
NEUROTENSOR_API Tensor extract_points_from_threshold(Tensor cloud, Scalar threshold, int64_t dims);
} // namespace tda
} // namespace nt

#endif
