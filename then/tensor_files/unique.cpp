#include "../cpu/unique.h"

namespace nt{
namespace functional{

Tensor unique(Tensor t, int64_t dim, bool return_unique,
              bool return_indices){
    return cpu::_unique(std::move(t), dim, return_unique, return_indices);
}

}
}
