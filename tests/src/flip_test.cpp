#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void flip_test(){
    using namespace nt::literals;
    // flip test
    run_test("flip all", [] {
        nt::Tensor a = nt::randn({5, 30, 30}, nt::DType::Float32);
        nt::Tensor f = nt::flip(a);
        nt::utils::throw_exception(nt::any(f != a), "Error, after flip a and the resulting tensor were the exact same");
    });
    run_test("flip dimensions", [] {
        nt::Tensor a = nt::randn({5, 30, 30}, nt::DType::Float32);
        nt::Tensor f = nt::flip(ntarg_(list) = {1}, ntarg_(input) = a);
        nt::utils::throw_exception(nt::any(f != a), "Error, after flip a and the resulting tensor were the exact same");
    });
    run_test("flip_view all", [] {
        nt::Tensor a = nt::randn({5, 30, 30}, nt::DType::Float32);
        nt::Tensor f = nt::flip_view(a);
        nt::utils::throw_exception(nt::any(f != a), "Error, after flip a and the resulting tensor were the exact same");
    });
    run_test("flip_view dimensions", [] {
        nt::Tensor a = nt::randn({5, 30, 30}, nt::DType::Float32);
        nt::Tensor f = nt::flip_view(ntarg_(list) = {1}, ntarg_(input) = a);
        nt::utils::throw_exception(nt::any(f != a), "Error, after flip a and the resulting tensor were the exact same");
    });

}

