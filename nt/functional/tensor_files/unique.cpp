#include <optional>
#include "../cpu/unique.h"

namespace nt{
namespace functional{

Tensor unique(Tensor t, std::optional<int64_t> dim, bool return_unique,
              bool return_indices){
    if(dim.has_value()){return cpu::_unique(std::move(t), dim.value(), return_unique, return_indices);}
    return cpu::_unique_vals_only(t, return_unique, return_indices);
}

}
}
