#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


int total_errors = 0;
int total_success = 0;

template<typename Func>
void run_mutable_test(const char* name, Func function){
    try{
        function();
        std::cout << "\033[1;31m[✗]\033[0m " << name << " Error: did not throw exception due to mutability"  << "\n";
        ++total_errors;
    }catch(const std::exception& e){ ++total_success; }
}

#define NT_SET_FIRST_TENSOR_UNMUTABILE(t, ...)\
    t.set_mutability(false);\
    nt::utils::throw_exception(t.is_mutable() == false, "Error, expected tensor to be unmutable for mutability test");

// #define NT_MAKE_MUTABILITY_TEST(func_name, ...)\
//     run_mutable_test(func_name, []{
//         NT_SET_FIRST_TENSOR_UNMUTABILE(__VA_ARGS__)
//     });


#define GENERAL_SIZE {3, 3}

#define MAKE_RANDOM nt::rand(0, 3, GENERAL_SIZE, nt::DType::Float32)
#define MAKE_NRANDOM nt::randn(GENERAL_SIZE)

#define NT_MAKE_GENERAL_MUTABILITY_TEST_EMPTY_1(func_name, ...)\
    run_mutable_test(#func_name, []{\
        nt::Tensor a = nt::rand(0, 10, GENERAL_SIZE);\
        NT_SET_FIRST_TENSOR_UNMUTABILE(a);\
        nt::func_name(a);\
    });

#define NT_MAKE_GENERAL_MUTABILITY_TEST_EMPTY_0(func_name, ...)\
    run_mutable_test(#func_name, []{\
        nt::Tensor a = nt::rand(0, 10, GENERAL_SIZE);\
        NT_SET_FIRST_TENSOR_UNMUTABILE(a);\
        nt::func_name(a, __VA_ARGS__);\
    });



#define NT_MAKE_GENERAL_MUTABILITY_TEST(func_name, ...) _NT_GLUE_(NT_MAKE_GENERAL_MUTABILITY_TEST_EMPTY_, _NT_IS_EMPTY_(__VA_ARGS__))(func_name, __VA_ARGS__)

#define NT_GENERAL_FUNCTIONS(_)\
    _(sigmoid_) \
    _(sqrt_) \
    _(invsqrt_) \
    _(abs_) \
    _(relu_)\
    _(gelu_)\
    _(silu_)\
    _(log_)\
    _(exp_)\
    _(pow_, exponent = 2)\
    _(fill_diagonal_, value = 3.0)\
    _(fill_, value = 3.0)\
    _(set_, nt::rand(0, 3, GENERAL_SIZE, nt::DType::Float32))\
    _(fused_multiply_add_, MAKE_RANDOM, MAKE_RANDOM)\
    _(fused_multiply_subtract_, MAKE_RANDOM, MAKE_RANDOM)\
    _(clamp_, min = 3.0)\
    _(init::xavier_uniform_)\
    _(add_, MAKE_RANDOM)\
    _(multiply_, MAKE_RANDOM)\
    _(subtract_, MAKE_RANDOM)\
    _(divide_, MAKE_RANDOM)\
    _(inverse_)\
    _(row_col_swap_)\
    _(tan_)\
    _(tanh_)\
    _(atan_)\
    _(atanh_)\
    _(cotan_)\
    _(cotanh_)\
    _(sin_)\
    _(sinh_)\
    _(asin_)\
    _(asinh_)\
    _(csc_)\
    _(csch_)\
    _(cos_)\
    _(cosh_)\
    _(acos_)\
    _(acosh_)\
    _(sec_)\
    _(sech_)\

void mutability_test(){
    using namespace nt::literals;
    std::cout << "\033[1;34m !-- TESTING MUTABILITY --! \033[0m\n" << std::endl;
    NT_GENERAL_FUNCTIONS(NT_MAKE_GENERAL_MUTABILITY_TEST)
    std::cout << "\033[1;34m \n !-- COMPLETED MUTABILITY TEST WITH "<<total_errors<<" ERRORS AND "<<total_success<<" SUCCESSES \033[0m\n" << std::endl;

}


#undef NT_SET_FIRST_TENSOR_UNMUTABILE
#undef NT_MAKE_GENERAL_MUTABILITY_TEST
#undef NT_MAKE_GENERAL_MUTABILITY_TEST_EMPTY_1 
#undef NT_MAKE_GENERAL_MUTABILITY_TEST_EMPTY_0 
#undef NT_GENERAL_FUNCTIONS 
#undef GENERAL_SIZE 
// #undef NT_MAKE_MUTABILITY_TEST 
