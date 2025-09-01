#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void sort_test(){
    using namespace nt::literals;
    // sort test
    run_test("sort - basic ascending", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor a = nt::arange(9, dtype = dt).view(3, 3).flip(1);  // reversed rows
            auto [sorted, indices] = nt::get<2>(nt::sort(a, dim = 1));
            nt::Tensor expected = nt::arange(9, dtype = dt).view(3, 3);
            nt::utils::throw_exception(nt::allclose(sorted, expected), "Sort failed for $ $ \n$ \n$", dt, nt::noprintdtype, expected.flatten(0, -1), sorted.view(-1));
        }
    });
    
    run_test("split_axis - check seg fault", []{
        nt::Tensor example({10, 10, 10}, nt::DType::Float32);
        nt::Tensor split = example.split_axis(0).view(-1);
        nt::Tensor sort({11}, nt::DType::int64);
        sort << 1, 2, 3, 4, 5, 6, 7, 9, 8, 0, 1;
        nt::Tensor sorted = split[sort];
        nt::Tensor catted = nt::cat(std::move(sorted));

    });

    run_test("coordsort - coordinate sort", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor a({3, 2}, dt);
            a << 3, 9,
                 1, 4,
                 2, 5;
            nt::Tensor out = nt::coordsort(a, dim=0);
            auto [sorted, indices] = nt::get<2>(out);
            nt::Tensor expected({3, 2}, dt); expected << 1, 4, 2, 5, 3, 9;
            nt::utils::throw_exception(nt::allclose(sorted, expected), "Coordsort failed for $ $ \n$ \n$", dt, nt::noprintdtype, expected.view(-1), sorted.view(-1));
        }
    });

}

