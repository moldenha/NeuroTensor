#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void unique_test(){
    using namespace nt::literals;
    run_test("unique - simple 1D", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor a(nt::SizeRef({6}), dt); a << 1, 2, 2, 3, 3, 3;
            auto [u, idx] = nt::get<2>(nt::unique(input = a));
            nt::Tensor expected(nt::SizeRef({3}), dt); expected << 1, 2, 3;
            nt::utils::throw_exception(nt::allclose(u, expected), "Unique values mismatch on dtype $ $ $ $", nt::noprintdtype, dt, u, expected);
        }
        nt::Tensor a(nt::SizeRef({6}), nt::DType::Bool); a << true, true, false, false, true, false;
        auto [u, idx] = nt::get<2>(nt::unique(input = a));
        nt::Tensor expected(nt::SizeRef({2}), nt::DType::Bool); expected << true, false;
        nt::utils::throw_exception(nt::all(nt::equal(u, expected)), "Unique values mismatch on dtype $ $ $ $", nt::noprintdtype, nt::DType::Bool, u, expected);
    });
    run_test("unique - multi ND", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor a(nt::SizeRef({5,6}), dt); 
            a << 1, 2, 2, 3, 3, 3,
                 3, 1, 3, 14, 1, 4,
                 1, 2, 2, 3, 3, 3,
                 4, 4, 4, 4, 4, 4, 
                 3, 1, 3, 14, 1, 4;
            auto u = nt::unique(a, dim = -1, return_indices = false);
            nt::Tensor expected(nt::SizeRef({3, 6}), dt); 
            expected <<  1, 2, 2, 3, 3, 3,
                         3, 1, 3, 14, 1, 4,
                         4, 4, 4, 4, 4, 4; 
            nt::utils::throw_exception(nt::allclose(u, expected), "Unique values mismatch for dtype $", dt);
        }
        nt::Tensor a(nt::SizeRef({5,6}), nt::DType::Bool); 
        a << true, true, false, false, true, false,
             true, false, true, false, true, false,
             true, true, false, false, true, false,
             false, false, false, false, false, false, 
             true, false, true, false, true, false;
        auto u = nt::unique(a, dim = -1, return_indices = false);
        nt::Tensor expected(nt::SizeRef({3, 6}), nt::DType::Bool); 
        expected <<  true, true, false, false, true, false,
                     true, false, true, false, true, false,
                     false, false, false, false, false, false; 
        nt::utils::throw_exception(nt::all(nt::equal(u, expected)), "Unique values mismatch for dtype $", nt::DType::Bool);


    });
 
}

