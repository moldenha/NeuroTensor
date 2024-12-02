#include <cstdint>

#include "../Tensor.h"
#include "../bucket/iterator.h"
#include "../refs/SizeRef.h"
#include "../dtype/ArrayVoid.h"
#include "../dtype/DType.h"
#include "../dtype/DType_enum.h"

#include <atomic>
#include <functional>
//#include <i386/types.h>
#include <memory.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <ratio>
#include <iterator>

#include <cassert>
//#include <format>
#include <sys/_types/_int32_t.h>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include <type_traits>
#include <vector>
#include "../utils/utils.h"
#include <chrono>
#include "../permute/permute_old.h"
#include "functional.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <random>
#include <cmath>
#include "../dtype/ArrayVoid.hpp"
#include "functional_operator.h"
#include "../mp/Threading.h"


namespace nt{
namespace functional{


//below is using intel's mkl library
#if defined(__x86_64__) || defined(__i386__)
#include "matmult_mkl.cpp"
#endif

}
}
