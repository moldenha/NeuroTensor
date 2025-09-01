#ifndef NT_PLOT_DIAGRAMS_H__
#define NT_PLOT_DIAGRAMS_H__

#include "../Tensor.h"
// #include "../../third_party/matplot/source/matplot/matplot.h"
#include <matplot/matplot.h>
#include <algorithm>
#include <tuple>
#include <vector>

namespace nt {
namespace tda {
NEUROTENSOR_API void plotPersistentDiagram(
    const std::vector<std::vector<std::tuple<Tensor, double, double>>>
        &homologyData);
NEUROTENSOR_API void plotBarcode(
    const std::vector<std::vector<std::tuple<Tensor, double, double>>>
        &homologyData);
NEUROTENSOR_API void plotPointCloud(Tensor cloud, int8_t point, int64_t dims);

} // namespace tda
} // namespace nt

#endif
