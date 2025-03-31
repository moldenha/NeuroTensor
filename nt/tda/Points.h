#ifndef _NT_TDA_POINTS_H_
#define _NT_TDA_POINTS_H_

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
Tensor extract_points_from_cloud(Tensor cloud, Scalar point, int64_t dims);
Tensor extract_points_from_threshold(Tensor cloud, Scalar threshold, int64_t dims);
} // namespace tda
} // namespace nt

#endif
