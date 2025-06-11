#include "../../Tensor.h"
#include "../../functional/functional.h"

namespace nt{
namespace fmri{

Tensor affine_transform(const Tensor &fmri, const Tensor &matrix) {
    // Get volume dimensions
    auto dims = fmri.shape();
    auto d = dims.size();
    
    utils::THROW_EXCEPTION(d == 3, "Affine Transformation: Expected 3D tensor, but got $D", d);

    int64_t x_dim = dims[0];
    int64_t y_dim = dims[1];
    int64_t z_dim = dims[2];

    // Create meshgrid coordinates
    Tensor x = nt::arange(0, x_dim, 1, fmri.dtype);
    Tensor y = nt::arange(0, y_dim, 1, fmri.dtype);
    Tensor z = nt::arange(0, z_dim, 1, fmri.dtype);

    Tensor xy = meshgrid(std::move(x), std::move(y));  // 2D grid
    Tensor xz = meshgrid(std::move(x), std::move(z));  // 2D grid
    Tensor yz = meshgrid(std::move(y), std::move(z));  // 2D grid

    // Apply 3D affine transformation
    Tensor grid = nt::cat({
        xy[0].reshape({-1, 1}),      // X-coordinates
        xy[1].reshape({-1, 1}),      // Y-coordinates
        yz[1].reshape({-1, 1}),      // Z-coordinates
        nt::functional::ones({xy[0].numel(), 1}, fmri.dtype)  // Homogeneous coords
    }, -1);
    

    // Apply affine matrix
    auto warped = nt::functional::matmult(grid, matrix, false, true);

    // Interpolate transformed grid
    return volume.grid_sample(warped);
}

}
}
