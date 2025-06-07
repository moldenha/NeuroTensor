#include "normalize.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid_NTensor.hpp"
#include "../../dtype/DType_enum.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"
#include <random>
#include "rand.h"

namespace nt {
namespace functional {
namespace cpu {

void xavier_uniform_(ArrayVoid &output, double bound) {
    rand_(output, Scalar(bound), Scalar(-bound)); //uniform real distribution
}

} // namespace cpu
} // namespace functional
} // namespace nt
