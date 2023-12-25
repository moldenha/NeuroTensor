#include "src/Tensor.h"
#include "src/dtype/DType.h"
#include "src/dtype/DType_enum.h"
#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <ios>
#include <numeric>
#include <ostream>
#include "test.h"
#include <immintrin.h>

int main(){
	/* current_test(pseudo_nn_simple_a); */
	current_test(unfold_layer_test, nt::functional::arange({1,200,256,256}));
	/* current_test(mat_mult_test); */
}
