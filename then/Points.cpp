#include "../functional/functional.h"
#include "Points.h"

namespace nt {
namespace tda {

Tensor extract_points_from_cloud(Tensor cloud, Scalar point, int64_t dims) {
    utils::throw_exception(
        cloud.dims() >= dims,
        "Expected to process cloud with dims of at least $ but got $", dims,
        cloud.dims());
    // if(cloud.dtype != DType::int8){cloud = cloud.to(DType::int8);}
    if (cloud.dims() == dims) {
        cloud = cloud.unsqueeze(0);
    } else if (cloud.dims() > (dims + 1)) {
        cloud = cloud.flatten(-1, dims + 1);
    }
    Tensor points_w =
        functional::where((cloud == point).split_axis((-1) * (dims + 1)));
    for (int64_t i = 0; i < points_w.shape()[0]; ++i) {
        points_w[i].item<Tensor>() =
            functional::stack(points_w[i].item<Tensor>());
    }
    Tensor points = points_w.RowColSwap_Tensors();
    return std::move(points);
}

Tensor extract_points_from_threshold(Tensor cloud, Scalar threshold,
                                     int64_t dims) {
    utils::throw_exception(
        cloud.dims() >= dims,
        "Expected to process cloud with dims of at least $ but got $", dims,
        cloud.dims());
    // if(cloud.dtype != DType::int8){cloud = cloud.to(DType::int8);}
    if (cloud.dims() == dims) {
        cloud = cloud.unsqueeze(0);
    } else if (cloud.dims() > (dims + 1)) {
        cloud = cloud.flatten(-1, dims + 1);
    }
    Tensor points_w =
        functional::where((cloud > threshold).split_axis((-1) * (dims + 1)));
    for (int64_t i = 0; i < points_w.shape()[0]; ++i) {
        points_w[i].item<Tensor>() =
            functional::stack(points_w[i].item<Tensor>());
    }
    Tensor points = points_w.RowColSwap_Tensors();
    return std::move(points);
}

} // namespace tda
} // namespace nt
