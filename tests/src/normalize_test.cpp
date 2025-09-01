#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void normalize_test(){
    using namespace nt::literals;
    // flip test
    run_test("xavier_uniform_", [] {

        nt::Tensor a = nt::rand(1, 10, {5, 30, 30}, nt::DType::Float32);
        nt::Tensor b = a.clone();
        nt::init::xavier_uniform_(a);
        nt::utils::throw_exception(nt::any(a != b), "Error, xavier uniform did nothing");
    });
    run_test("var", [] {
        nt::Tensor a({4, 4}, nt::DType::Float32);
        a << 0.2035,  1.2959,  1.8101, -0.4644,
             1.5027, -0.3270,  0.5905,  0.6538,
            -1.5745,  1.3330, -0.5596, -0.6548,
             0.1264, -0.5080,  1.6420,  0.1992;
        nt::Tensor check({4, 1}, nt::DType::Float32);
        check << 1.06308, 0.559027, 1.48931, 0.825759;
        nt::Tensor out = nt::var(a, dim = 1, keepdim = true);
        nt::utils::throw_exception(out.shape() == check.shape(), "Error, var did not return the correct shape");
        nt::utils::throw_exception(nt::allclose(out, check), "Error, tensors did not match from variance");
    });

}

