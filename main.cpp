#include "src/Tensor.h"
#include "src/dtype/DType.h"
#include "src/dtype/DType_enum.h"
#include <_types/_uint32_t.h>
#include <ios>
#include <numeric>
#include <ostream>
/* #include "autograd_test.h" */
#include "src/dtype/ArrayVoid.hpp"
#include "src/mp/Threading.h" //testing this at the moment
/* #include "tda_test.h" */
#include <chrono>
#include <random>

#include <immintrin.h>
#include <mutex>
#include "tests/tensor_test.h"
// #include "tests/tensorgrad_test.h"
// #include "tests/layer_test.h"





int main(){
	convT_gradient_tests();
	/* test_layers(); */
	// test_lnn();

	return 0;
}

