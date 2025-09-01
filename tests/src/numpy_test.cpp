#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void numpy_test(){
    using namespace nt::literals;
    // flip test
    run_test("numpy_save", [] {
        nt::Tensor a({4, 4}, nt::DType::Float32);
        a << 0.2035,  1.2959,  1.8101, -0.4644,
             1.5027, -0.3270,  0.5905,  0.6538,
            -1.5745,  1.3330, -0.5596, -0.6548,
             0.1264, -0.5080,  1.6420,  0.1992;
        nt::to_numpy(str = "example.npy", input = a);
    });
    run_test("numpy_load", [] {
        nt::Tensor a({4, 4}, nt::DType::Float32);
        a << 0.2035,  1.2959,  1.8101, -0.4644,
             1.5027, -0.3270,  0.5905,  0.6538,
            -1.5745,  1.3330, -0.5596, -0.6548,
             0.1264, -0.5080,  1.6420,  0.1992;
        nt::Tensor out = nt::from_numpy(str = "example.npy");
        nt::utils::throw_exception(nt::allclose(out, a), "Error, tensors did not match from loading back from numpy");
    });

}

