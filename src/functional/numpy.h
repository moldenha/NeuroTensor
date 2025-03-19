#ifndef _NT_NUMPY_CONVERSION_H_
#define _NT_NUMPY_CONVERSION_H_
#include "../Tensor.h"
#include <string>

namespace nt {
namespace functional {
Tensor from_numpy(std::string);
void to_numpy(const Tensor &, std::string);

} // namespace functional
} // namespace nt

#endif
