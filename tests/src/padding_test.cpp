#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void padding_test(){
    using namespace nt::literals;
    // padding test
    run_test("pad", [] {
        nt::Tensor a({3, 3}, nt::DType::Float32);
        a << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;
        nt::Tensor out = nt::pad(a, padding = {1, 0, 1, 1});
        
        nt::Tensor check({4, 5}, nt::DType::Float32);
        check << 0, 0, 0, 0, 0,
                 0, 1, 2, 3, 0,
                 0, 4, 5, 6, 0,
                 0, 7, 8, 9, 0;

        nt::utils::throw_exception(nt::all(out == check), "Error, pad did not work");
    });
    run_test("unpad", [] {
        nt::Tensor a({4, 5}, nt::DType::Float32);
        a << 0, 0, 0, 0, 0,
             0, 1, 2, 3, 0,
             0, 4, 5, 6, 0,
             0, 7, 8, 9, 0;
        nt::Tensor out = nt::unpad(a, padding = {1, 0, 1, 1});
        
        nt::Tensor check({3, 3}, nt::DType::Float32);
        check << 1, 2, 3,
                 4, 5, 6,
                 7, 8, 9;

        nt::utils::throw_exception(nt::all(out == check), "Error, pad did not work");
    });
}

