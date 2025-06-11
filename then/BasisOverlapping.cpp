#include "BasisOverlapping.h"

namespace nt {
namespace tda {

BasisOverlapping::BasisOverlapping(Tensor points) {
    utils::throw_exception(points.dtype == DType::TensorObj,
                           "Expected to get batches of tensors in terms of "
                           "tensor objects but got $",
                           points.dtype);
    utils::throw_exception(points[0].item<Tensor>().dtype == DType::int64,
                           "Expected to get the coordinates of the points in "
                           "terms of int64, but got $",
                           points[0].item<Tensor>().dtype);
    utils::throw_exception(
        points[0].item<Tensor>().dims() == 2,
        "Expected to get the dims of the points as 2, but got $",
        points[0].item<Tensor>().dims());

    int D = points[0].item<Tensor>().shape()[1];
    if (points.numel() == 1) {
        // no need to batch
        Tensor &pts = points.item<Tensor>();
        Tensor diff = pts.view(-1, 1, D) - pts.view(1, -1, D);
        this->dist_sq = Tensor::makeNullTensorArray(1);
        this->dist_sq.item<Tensor>() = diff.pow(2).sum(-1).to(DType::Float64);
        return;
    }
    Tensor cpy_pts = Tensor::makeNullTensorArray(points.numel());
    // Tensor cpy_pts = points.view_Tensors(-1, 1, D).transpose_Tensors(-1, -2);
    for (int64_t i = 0; i < cpy_pts.numel(); ++i) {
        auto sh = points[i].item<Tensor>().shape();
        cpy_pts[i] = points[i]
                         .item<Tensor>()
                         .view(1, -1, D)
                         .expand({sh[0], sh[0], D})
                         .clone();
        // std::cout << "cpy_pts[i]: "<<cpy_pts[i].item<Tensor>() << std::endl;
    }
    // Expand points: (N, 1, D) - (1, N, D) -> (N, N, D)
    Tensor diff = points.view_Tensors(-1, 1, D) - cpy_pts;

    // Compute squared Euclidean distances: sum across last dimension (D)
    this->dist_sq = diff.pow(2).sum(-1);
    Tensor* begin = reinterpret_cast<Tensor*>(this->dist_sq.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(this->dist_sq.data_ptr_end());
    for(;begin != end; ++begin)
        *begin = begin->to(DType::Float64);
}

Tensor BasisOverlapping::adjust_radius(double r) const {
    // Compute squared radius sum (r + r)^2 = (2r)^2
    double r_sq = 4 * (r * r);

    // Compute mask: dist_sq <= (2r)^2
    Tensor overlap_mask = (dist_sq <= r_sq);

    // Remove self-overlap (diagonal elements)
    overlap_mask.fill_diagonal_(false);

    // Return indices of overlapping pairs
    // return overlap_mask.nonzero();
    return std::move(overlap_mask);
}

//multiplies dist_sq *= std::pow(weight, 2)
void BasisOverlapping::add_weight(Tensor weight){
    dist_sq[0].item<Tensor>() *= std::pow(weight, 2);
}

} // namespace tda
} // namespace nt
