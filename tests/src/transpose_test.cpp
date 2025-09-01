#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>

void transpose_test(){
    using namespace nt::literals;
    // transpose test
    run_test("permute - swap dims", [] {
        nt::Tensor a = nt::arange(6).view(2, 3);
        auto b = nt::permute(input = a, dims = {1, 0});
        nt::utils::throw_exception(b.shape() == nt::SizeRef({3, 2}), "Permute failed");
    });

    run_test("transpose - 2D", [] {
        nt::Tensor a = nt::arange(4).view(2, 2);
        auto b = nt::transpose(input = a, dim0 = 0, dim1 = 1);
        nt::utils::throw_exception(nt::allclose(a, b.transpose(1, 0)), "Transpose failed");
    });

    run_test("row_col_swap_ - symmetric matrix", [] {
        nt::Tensor a = nt::arange(4).view(2, 2);
        nt::row_col_swap_(input = a);
        nt::Tensor expected({2, 2}); expected << 0, 2, 1, 3;
        nt::utils::throw_exception(nt::allclose(a, expected), "row_col_swap_ failed");
    });

}

