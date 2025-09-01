#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void mesh_test(){
    using namespace nt::literals;
    // flip test
    run_test("one_hot a", [] {
        nt::Tensor i = nt::arange(5, dtype = nt::DType::int64) % 3;
        nt::Tensor o = nt::one_hot(i);
        nt::Tensor expected({5, 3}, nt::DType::int64);
        expected << 1, 0, 0,
                    0, 1, 0,
                    0, 0, 1,
                    1, 0, 0,
                    0, 1, 0;

        nt::utils::throw_exception(nt::all(o == expected), "Error got wrong output ($)", o.shape());
    });
    run_test("one_hot b", [] {
        nt::Tensor i = nt::arange(5, dtype = nt::DType::int64) % 3;
        nt::Tensor o = nt::one_hot(i, num_classes = 5);
        nt::Tensor expected({5, 5}, nt::DType::int64);
        expected << 1, 0, 0, 0, 0,
                    0, 1, 0, 0, 0,
                    0, 0, 1, 0, 0,
                    1, 0, 0, 0, 0,
                    0, 1, 0, 0, 0;
        nt::utils::throw_exception(nt::all(o == expected), "Error got wrong output ($)", o.shape());
    });
    run_test("one_hot c", [] {
        nt::Tensor i = nt::arange(dtype = nt::DType::int64, size = {3, 2}) % 3;
        nt::Tensor o = nt::one_hot(i);
        nt::Tensor expected({3, 2, 3}, nt::DType::int64);
        expected << 1, 0, 0,
                    0, 1, 0,

                    0, 0, 1,
                    1, 0, 0,
                    
                    0, 1, 0,
                    0, 0, 1;
        nt::utils::throw_exception(nt::all(o == expected), "Error got wrong output ($)", o.shape());
    });
    run_test("mesh grid", [] {
        nt::Tensor a({3}, nt::DType::int64);
        a << 1, 2, 3;
        nt::Tensor b({3}, nt::DType::int64);
        b << 4, 5, 6;
       
        nt::Tensor expected_x({3, 3}, nt::DType::int64);
        expected_x << 1, 1, 1,
                      2, 2, 2,
                      3, 3, 3;

        nt::Tensor expected_y({3, 3}, nt::DType::int64);
        expected_y << 4, 5, 6,
                      4, 5, 6,
                      4, 5, 6;
        auto [grid_x, grid_y] = nt::get<2>(nt::meshgrid(a, b));
        nt::utils::throw_exception(nt::all(grid_x == expected_x), "Error, got wrong output for x grid ($) ($)", grid_x.shape(), expected_x.shape());
        nt::utils::throw_exception(nt::all(grid_y == expected_y), "Error, got wrong output for y grid ($) ($)", grid_y.shape(), expected_y.shape());
    });

}

