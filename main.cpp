#include "src/Tensor.h"
#include "src/dtype/DType.h"
#include "src/dtype/DType_enum.h"
#include <ios>
#include <numeric>
#include <ostream>
#include "test.h"

int main(){
	/* current_test(pseudo_nn_simple_a); */
	current_test(unfold_layer_test, nt::functional::arange({1,2,8,8}));
	/* current_test(mat_mult_test); */
}
