#ifndef _FUNCTIONAL_MATMULT_H_
#define _FUNCTIONAL_MATMULT_H_

#include "../../Tensor.h"
#include <cstddef>

//look into a libbf16
namespace nt{
namespace functional{

Tensor matmult(const Tensor&, const Tensor&, bool trans_a = false, bool trans_b = false);
Tensor matmult_cT(const Tensor&, const Tensor&);

}

}

#endif
