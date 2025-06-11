#ifndef _NT_TDA_BASIS_OVERLAPPING_H_
#define _NT_TDA_BASIS_OVERLAPPING_H_

#include "../Tensor.h"
#include "../convert/std_convert.h"
#include "../functional/functional.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace nt {
namespace tda {

class BasisOverlapping {
    Tensor dist_sq;

  public:
    BasisOverlapping() = default;
    BasisOverlapping(Tensor points);
    Tensor adjust_radius(double r) const;
    //multiplies dist_sq *= std::pow(weight, 2)
    void add_weight(Tensor weight);
    inline Tensor get_distances(int64_t batch) const {
        Tensor t = dist_sq[batch].item<Tensor>().to(DType::Float64);
        return t;
    }
    inline size_t num_points(int64_t batch) const noexcept {
        return static_cast<size_t>(dist_sq[batch].item<Tensor>().shape()[0]);
    }
    inline bool is_batched() const noexcept { return dist_sq.numel() > 1; }
};

} // namespace tda
} // namespace nt

#endif //_NT_TDA_BASIS_OVERLAPPING_H_
