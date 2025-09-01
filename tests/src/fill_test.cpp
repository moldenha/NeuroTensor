#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void fill_test(){
    using namespace nt::literals;
    // fill test
    run_test("zeros", [] {
        nt::Tensor a = nt::zeros({10, 10}, ntarg_(dtype) = nt::DType::Float32);
        nt::utils::throw_exception(nt::all(a == 0), "Error zeros was not all equal to zeros");
    });
    run_test("ones", [] {
        nt::Tensor a = nt::ones({10, 10}, ntarg_(dtype) = nt::DType::Float32);
        nt::utils::throw_exception(nt::all(a == 1), "Error ones was not all equal to ones");
    });
    run_test("nums (4)", [] {
        nt::Tensor a = nt::nums(ntarg_(num) = 4, ntarg_(size) = {10, 10}, ntarg_(dtype) = nt::DType::Float32);
        nt::utils::throw_exception(nt::all(a == 4), "Error nums was not all equal to 4");
    });
    run_test("arange", [] {
        nt::Tensor a = nt::arange({10, 10}, ntarg_(start) = nt::complex_64(1), ntarg_(dtype) = nt::DType::Float32);
        nt::utils::throw_exception(nt::none(a == 0), "Error arange did not start correctly");
    });

    run_test("fill_diagonal_", [] {
        nt::Tensor a = nt::randn({10, 10}, nt::DType::Float32);
        nt::utils::throw_exception(nt::none(a == 20), "Error randn has a variable well above 1");
        nt::fill_diagonal_(ntarg_(input) = a, ntarg_(value) = 20);
        nt::utils::throw_exception(nt::all(nt::functional::diagonal(a) == 20), "Error, expected a diagonal to be all 20");
    });

    run_test("set_", []{
        nt::Tensor a = nt::rand(0, 10, {10, 10}, nt::DType::Float32);
        nt::Tensor b = nt::rand(20, 30, {10, 10}, nt::DType::Float32);
        nt::utils::throw_exception(!nt::all(a == b), "Error, rand created 2 identical tensors which should not have been possible");
        nt::set_(ntarg_(input) = a, ntarg_(tensor) = b);
        nt::utils::throw_exception(nt::all(a == b), "Error, set did not set all tensors equal");
    });
}

