#ifndef NT_FUNCTIONAL_TENSOR_FILES_PRINT_H__
#define NT_FUNCTIONAL_TENSOR_FILES_PRINT_H__

#include <iostream>
#include <ostream>
#include "../../Tensor.h"

namespace nt{
namespace functional{

NEUROTENSOR_API std::ostream& print(std::ostream&, const Tensor&);
NEUROTENSOR_API void print(const Tensor&);

}
}

#endif
