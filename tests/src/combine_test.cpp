#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void combine_test(){
    using namespace nt::literals;
    // cat test
    run_test("cat", [] {
        nt::Tensor a = nt::ones({1, 5}, nt::DType::Float32);
        nt::Tensor b = nt::zeros({2, 5}, nt::DType::Float32);
        nt::Tensor c = nt::nums({1, 5}, 3, nt::DType::Float32);
        auto y = nt::cat({a, b, c}, dim = 0);
        nt::utils::throw_exception(!a.is_null(), "Error a was made null inside of the initializer list for cat");
        nt::utils::throw_exception(!b.is_null(), "Error b was made null inside of the initializer list for cat");
        nt::utils::throw_exception(!c.is_null(), "Error c was made null inside of the initializer list for cat");
        
        nt::utils::throw_exception(y.dtype() == nt::DType::Float32, "Output dtype should be float32");
        nt::utils::throw_exception(y.shape() == nt::SizeRef({4, 5}), "Error cat output shape should be {4,5}, got $", y.shape());
        y.arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Float32> > >(
         [](auto begin, auto end){
            int64_t i = 0;
            for(;i < 5; ++i, ++begin){
                nt::utils::throw_exception(*begin == 1.0f, "Error, catted wrong at ones");
            }
            for(;i < 15; ++i, ++begin){
                nt::utils::throw_exception(*begin == 0.0f, "Error, catted wrong at zeros");
            }
            for(;i < 20; ++i, ++begin){
                nt::utils::throw_exception(*begin == 3.0f, "Error, catted wrong at threes");
            }
        });
    });
    run_test("stack", [] {
        nt::Tensor a = nt::ones({5}, nt::DType::Float32);
        nt::Tensor b = nt::zeros({5}, nt::DType::Float32);
        nt::Tensor c = nt::nums({5}, 3, nt::DType::Float32);
        nt::utils::throw_exception(!a.is_null(), "Error a was made null inside of the initializer list for stack");
        nt::utils::throw_exception(!b.is_null(), "Error b was made null inside of the initializer list for stack");
        nt::utils::throw_exception(!c.is_null(), "Error c was made null inside of the initializer list for stack");
        auto y = nt::stack({a, b, c}, dim = 0);
        nt::utils::throw_exception(y.dtype() == nt::DType::Float32, "Output dtype should be float32");
        nt::utils::throw_exception(y.shape() == nt::SizeRef({3, 5}), "Error stack output shape should be {3,5}, got $", y.shape());
        y.arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Float32> > >(
         [](auto begin, auto end){
            int64_t i = 0;
            for(;i < 5; ++i, ++begin){
                nt::utils::throw_exception(*begin == 1.0f, "Error, catted wrong at ones");
            }
            for(;i < 10; ++i, ++begin){
                nt::utils::throw_exception(*begin == 0.0f, "Error, catted wrong at zeros");
            }
            for(;i < 15; ++i, ++begin){
                nt::utils::throw_exception(*begin == 3.0f, "Error, catted wrong at threes");
            }
        });
    });

}

