#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void repeat_test(){
    using namespace nt::literals;
    // repeat test
    run_test("repeat - flatten repeat", [] {
        nt::Tensor x({2}, nt::DType::Float32);
        x << 1.0f, 2.0f;

        nt::Tensor expected({4}, nt::DType::Float32);
        expected << 1.0f, 2.0f, 1.0f, 2.0f;

        auto out = nt::repeat(x, amt = 2);

        nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch");
    });
    run_test("repeat - repeat along dim", [] {
        nt::Tensor x({2, 1}, nt::DType::Float32);
        x << 3.0f, 4.0f;

        nt::Tensor expected({2, 3}, nt::DType::Float32);
        expected << 3.0f, 3.0f, 3.0f,
                    4.0f, 4.0f, 4.0f;

        auto out = nt::repeat(x, dim = 1, amt = 3);  // repeat 3 times along dim 1

        nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch");
    });
    run_test("repeat - Tensors", [] {
        nt::Tensor x = nt::rand(0, 3, {4, 5, 8, 8});
        nt::Tensor y = x.split_axis(-3);
        nt::Tensor z = y.repeat_(10);
        // y = y.repeat_(10);
    });
    run_test("repeat - repeat along dim", [] {
        nt::Tensor x({2, 1}, nt::DType::Float32);
        x << 3.0f, 4.0f;

        nt::Tensor expected({2, 3}, nt::DType::Float32);
        expected << 3.0f, 3.0f, 3.0f,
                    4.0f, 4.0f, 4.0f;

        auto out = nt::repeat(x, dim = 1, amt = 3);  // repeat 3 times along dim 1

        nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch");
    });
    run_test("expand - broadcast to shape", [] {
        nt::Tensor x({1, 3}, nt::DType::Float32);
        x << 1.0f, 2.0f, 3.0f;

        nt::Tensor expected({2, 3}, nt::DType::Float32);
        expected << 1.0f, 2.0f, 3.0f,
                    1.0f, 2.0f, 3.0f;

        auto out = nt::expand(x, {2, 3});

        nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch");
    });
    run_test("expand_as - broadcast to match tensor", [] {
        nt::Tensor x({1, 3}, nt::DType::Float32);
        x << 5.0f, 6.0f, 7.0f;

        nt::Tensor other = nt::zeros(size = {2, 3}, dtype = nt::DType::Float32);  // shape target

        nt::Tensor expected({2, 3}, nt::DType::Float32);
        expected << 5.0f, 6.0f, 7.0f,
                    5.0f, 6.0f, 7.0f;

        auto out = nt::expand_as(x, other);

        nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch");
    });
}

