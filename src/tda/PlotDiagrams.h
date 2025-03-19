#ifndef _NT_PLOT_DIAGRAMS_H_
#define _NT_PLOT_DIAGRAMS_H_

#include "../Tensor.h"
#include "matplot/matplot.h"
#include <algorithm>
#include <tuple>
#include <vector>

namespace nt {
namespace tda {
void plotPersistentDiagram(
    const std::vector<std::vector<std::tuple<Tensor, double, double>>>
        &homologyData);
void plotBarcode(
    const std::vector<std::vector<std::tuple<Tensor, double, double>>>
        &homologyData);
void plotPointCloud(Tensor cloud, int8_t point, int64_t dims);

} // namespace tda
} // namespace nt

#endif
