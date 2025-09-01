#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void split_test(){
    using namespace nt::literals;
    // split test
    run_test("split - manual sizes", [] {
        nt::Tensor a = nt::arange(6);
        auto outs = nt::split(a, 0, {2, 3, 1});
        nt::utils::throw_exception(outs[0].item<nt::Tensor>().shape()[0] == 2 && outs.numel() == 3, "Split shape mismatch");
    });

    run_test("chunk - even chunks", [] {
        nt::Tensor a = nt::arange(6);
        auto outs = nt::chunk(input = a, chunks = 3, dim = 0);
        nt::utils::throw_exception(outs.numel() == 3 && outs[0].item<nt::Tensor>().shape()[0] == 2, "Chunk split mismatch");
    });


}

