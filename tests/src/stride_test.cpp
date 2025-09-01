#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>
#include <nt/linalg/linalg.h>

void stride_test(){
    using namespace nt::literals;
    // stride test
    run_test("diagonal - square matrix", [] {
        nt::Tensor a = nt::linalg::eye(3);
        auto diag = nt::diagonal(input = a);
        nt::Tensor check = nt::ones({3});
        nt::utils::throw_exception(nt::allclose(diag, check), "Diagonal mismatch");
    });

    run_test("as_strided - reshape and stride", [] {
        nt::Tensor a = nt::arange(4).view(2, 2);
        auto view = nt::as_strided(input = a, size = {2, 2}, stride = {1, 2});
        nt::Tensor expected({2, 2}); expected << 0, 2, 1, 3;
        nt::utils::throw_exception(nt::allclose(view, expected), "as_strided mismatch");
    });
}

