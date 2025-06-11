#include "../../Tensor.h"
#include <string>

namespace nt{
namespace functional{
Tensor load(std::string);
void save(Tensor, std::string);
void save(const Tensor &, const char *);
Tensor load(const char *);
}
}
