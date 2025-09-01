// #include "tests/tensor_test.h"
// #include "tests/tensorgrad_test.h"
// #include "tests/layer_test.h"
// #include "tests/tda_test.h"
#include "tests/nn_tda_test.h"
// #include "tests/linalg_test.h"
// #include "tests/fmri_test.h"

#include "tests/pooling.h"

int main(){
    // svd_test();
    // qr_test();
    // inv_test();
    // pinv_test();
    // persistent_diagram_test();
	// convT_gradient_tests();
	/* test_layers(); */
	// test_lnn();
    // operator_test();
    // fmri_load();
    // eye_test();
    // nn_laplacian_2_test();
    // nn_laplacian_2_test_sub();
    nn_boundary_test();
    // row_swap_test();
    // softmax_test();
    // auto func1 = [](const nt::TensorGrad& x){return nt::functional::relu(x);}; 
    // auto func2 = [](const nt::TensorGrad& x){return nt::functional::softmax(x);}; 
    // bool worked = activation_function_test(func1, func2);
    // linear_test();
    // bool worked = test_gumbel_softmax_activation();
    // std::cout << std::boolalpha << "worked: "<<worked<<std::noboolalpha << std::endl;
    // symmetric_mult_test();
    // fractional_max_pool(2);

    return 0;
}
