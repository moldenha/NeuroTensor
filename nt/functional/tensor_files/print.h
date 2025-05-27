#ifndef __NT_FUNCTIONAL_TENSOR_FILES_PRINT_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_PRINT_H__

#include <iostream>
#include <ostream>
#include "../../Tensor.h"

namespace nt{
namespace functional{

std::ostream& print(std::ostream&, const Tensor&);
void print(const Tensor&);

}
}

#endif
