#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"



void combinations_test(){
    using namespace nt::literals;
    // unfold1d test
    run_test("combinations", [] {
        nt::Tensor x = nt::randn({16});
        auto y = nt::combinations(r = 3, vec = x);
    });
}

