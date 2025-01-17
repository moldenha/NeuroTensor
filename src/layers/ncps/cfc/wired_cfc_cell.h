#ifndef _NT_LAYERS_NCPS_WIRED_CFC_CELL_H_
#define _NT_LAYERS_NCPS_WIRED_CFC_CELL_H_

#include "../../../utils/type_traits.h"
#include "../../Layer.h"
#include "../../Module.h"
#include "../../layers.h"
#include "../wiring/wiring.h"
#include "cfc_cell.h"

namespace nt {
namespace ncps {

class WiredCfCCell : public Module {
    intrusive_ptr<Wiring> _wiring;
    std::vector<Layer> _layers;
    int64_t num_layers;
    std::vector<int64_t> layer_sizes(); 

  public:
    WiredCfCCell(int64_t input_size, intrusive_ptr<Wiring> wiring,
                 std::string mode = "default");
    WiredCfCCell(const WiredCfCCell&);
    WiredCfCCell(WiredCfCCell&&);
    WiredCfCCell& operator=(const WiredCfCCell&);
    WiredCfCCell& operator=(WiredCfCCell&&);

    TensorGrad forward(TensorGrad input, TensorGrad hx, Tensor timespans, TensorGrad& hx_out);
};

} // namespace ncps
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::ncps::WiredCfCCell, nt__ncps__WiredCfCCell)

#endif // _NT_LAYERS_NCPS_WIRED_CFC_CELL_H_
