#include "activation_func.h"
#include "col_im_tests.h"
#include "combinations_test.h"
#include "combine_test.h"
#include "compare_test.h"
#include "complex_test.h"
#include "conv_test.h"
#include "convert_test.h"
#include "dilate_test.h"
#include "dropout_test.h"
#include "fill_test.h"
#include "flip_test.h"
#include "index_test.h"
#include "matmult_test.h"
#include "mesh_test.h"
#include "min_max_test.h"
#include "normalize_test.h"
#include "numpy_test.h"
#include "operator_test.h"
#include "padding_test.h"
#include "pooling_test.h"
#include "range_test.h"
#include "round_test.h"
#include "repeat_test.h"
#include "save_load_test.h"
#include "softmax_test.h"
#include "sort_test.h"
#include "split_test.h"
#include "stride_test.h"
#include "sum_exp_log_test.h"
#include "transpose_test.h"
#include "trig_test.h"
#include "unique_test.h"
#include "intrusive_ptr_test.h"
#include "null_test.h"
#include "mutability_test.h"
#include "pytorch_test.h"
#include "tensor_grad_test.h"
#include "activation_func_autograd_test.h"
#include "col_im_autograd_test.h" 
#include "conv_autograd_test.h"
#include "linear_autograd_test.h"
#include "trig_autograd_test.h"
#include "min_max_autograd_test.h"
#include "autograd_view_test.h"
#include "operator_autograd_test.h"
#include "normalize_autograd_test.h"
// Autograd tests to back-test against pytorch
// While all of the functions are used for both nt::Tensor's and nt::TensorGrad's
// The following just have functions with non-trivial autograds that need to be back-tested
// - activation_test [done]
// - col_im_test [done]
// - matmult_test <- in linear_autograd_test [done]
// - conv_test [done]
// - dilate_test [no pytorch equivalent]
// - min_max_test [done]
// - normalize_test
// - operator_test [done]
// - padding_test
// - pooling_test
// - round_test
// - softmax_test
// - sort_test
// - split_test
// - sum_exp_log_test
// - transpose_test <- more of a view grad test
// - trig_test [done]
// - unique_test <- more of a view grad test


int main(){
    activation_test();
    col_im_test();
    combinations_test();
    combine_test();
    compare_test();
    complex_test();
    conv_test();
    convert_test();
    dilate_test();
    dropout_test();
    fill_test();
    flip_test();
    index_test();
    matmult_test();
    mesh_test();
    min_max_test();
    normalize_test();
    numpy_test();
    operator_test();
    padding_test();
    pooling_test();
    range_test();
    repeat_test();
    round_test();
    save_load_test();
    softmax_test();
    sort_test();
    split_test();
    stride_test();
    sum_exp_log_test();
    transpose_test();
    trig_test();
    unique_test();
    intrusive_ptr_test();
    null_test();
    mutability_test();
    // Testing autograd against the pytorch autograd
    // pytorch_test();
    tensor_grad_test();
    activation_test_autograd();
    col_im_test_autograd();
    linear_autograd_test();
    trig_test_autograd();
    conv_autograd_test();
    min_max_autograd_test();
    view_autograd_test();
    operator_autograd_test();
    normalize_autograd_test();
    return 0;
}
