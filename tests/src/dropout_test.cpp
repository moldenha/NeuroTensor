#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void dropout_test(){
    using namespace nt::literals;
    // dropout test
    run_test("dropout", [] {
        nt::Tensor a = nt::randn({30, 30, 30, 30}, nt::DType::Float32);
        nt::utils::throw_exception(!nt::any(a == 0), "Error randn returned 0's in the tensor");
        nt::Tensor d = nt::dropout(a, ratio = 0.4);
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error dropout returned only 0");
        nt::utils::throw_exception(nt::functional::any(d == 0), "Error dropout had no 0's");
    });
    run_test("dropout2d", [] {
        nt::Tensor a = nt::randn({30, 30, 30, 30}, nt::DType::Float32);
        nt::utils::throw_exception(!nt::any(a == 0), "Error randn returned 0's in the tensor");
        nt::Tensor d = nt::dropout2d(a, ratio = 0.4);
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error dropout returned only 0");
        nt::utils::throw_exception(nt::functional::any(d == 0), "Error dropout had no 0's");
    });
    run_test("dropout3d", [] {
        nt::Tensor a = nt::randn({30, 30, 30, 30}, nt::DType::Float32);
        nt::utils::throw_exception(!nt::any(a == 0), "Error randn returned 0's in the tensor");
        nt::Tensor d = nt::dropout3d(a, ratio = 0.4);
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error dropout returned only 0");
        nt::utils::throw_exception(nt::functional::any(d == 0), "Error dropout had no 0's");
    });
}

