#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void pooling_test(){
    using namespace nt::literals;
    run_test("avg_pool1d - basic", [] {
        nt::Tensor input({1, 1, 5}, nt::DType::Float32);
        input << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f;

        nt::Tensor check({1, 1, 2}, nt::DType::Float32);
        check << 1.5f, 3.5f;

        auto out = nt::avg_pool1d(input = input, kernel_size = 2, stride = 2);
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Value mismatch");
    });

    run_test("adaptive_avg_pool1d - basic", [] {
        nt::Tensor input({1, 1, 4}, nt::DType::Float32);
        input << 1.0f, 2.0f, 3.0f, 4.0f;

        nt::Tensor check({1, 1, 2}, nt::DType::Float32);
        check << 1.5f, 3.5f;

        auto out = nt::adaptive_avg_pool1d(input = input, output_size = 2);
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Value mismatch");
    });

    run_test("avg_pool2d - basic", [] {
        nt::Tensor input({1, 1, 2, 2}, nt::DType::Float32);
        input << 1.0f, 2.0f,
                 3.0f, 4.0f;

        nt::Tensor check({1, 1, 1, 1}, nt::DType::Float32);
        check << 2.5f;

        auto out = nt::avg_pool2d(input = input, kernel_size = {2, 2});
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Value mismatch");
    });

    run_test("adaptive_avg_pool2d - basic", [] {
        nt::Tensor input({1, 1, 2, 2}, nt::DType::Float32);
        input << 1.0f, 2.0f,
                 3.0f, 4.0f;

        nt::Tensor check({1, 1, 1, 1}, nt::DType::Float32);
        check << 2.5f;

        auto out = nt::adaptive_avg_pool2d(input = input, output_size = {1, 1});
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Value mismatch");
    });

    run_test("avg_pool3d - basic", [] {
        nt::Tensor input({1, 1, 2, 2, 2}, nt::DType::Float32);
        input << 1.0f, 2.0f,
                 3.0f, 4.0f,
                 5.0f, 6.0f,
                 7.0f, 8.0f;

        nt::Tensor check({1, 1, 1, 1, 1}, nt::DType::Float32);
        check << 4.5f;

        auto out = nt::avg_pool3d(input = input, kernel_size = {2, 2, 2});
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Value mismatch");
    });


    run_test("lp_pool1d - p=2", [] {
        nt::Tensor input({1, 1, 4}, nt::DType::Float32);
        input << 1.0f, 2.0f, 3.0f, 4.0f;

        nt::Tensor check({1, 1, 2}, nt::DType::Float32);
        check << std::sqrt(1*1 + 2*2), std::sqrt(3*3 + 4*4);  // ≈ 2.236, 5.0

        auto out = nt::lp_pool1d(input = input, kernel_size = 2, power = 2.0f);
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Value mismatch");
    });
    run_test("adaptive_lp_pool3d - basic", [] {
        nt::Tensor input({1, 1, 2, 2, 2}, nt::DType::Float32);
        input << 1.0f, 2.0f,
                 3.0f, 4.0f,
                 5.0f, 6.0f,
                 7.0f, 8.0f;

        float p = 2.0f;
        float expected = std::sqrt(1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8);  // ≈ 14.2829

        nt::Tensor check({1, 1, 1, 1, 1}, nt::DType::Float32);
        check << expected;

        auto out = nt::adaptive_lp_pool3d(input = input, output_size = {1, 1, 1}, power = p);
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-4), "Value mismatch");
    });

    run_test("max_pool1d - basic", [] {
        nt::Tensor input({1, 1, 4}, nt::DType::Float32);
        input << 1.0f, 3.0f, 2.0f, 4.0f;

        nt::Tensor check({1, 1, 2}, nt::DType::Float32);
        check << 3.0f, 4.0f;

        auto out = nt::max_pool1d(input = input, kernel_size = 2, stride = 2);
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch got $ expected $", out.shape(), check.shape());
        nt::utils::throw_exception(nt::allclose(out, check), "Value mismatch");
    });

    run_test("adaptive_max_pool2d - basic", [] {
        nt::Tensor input({1, 1, 2, 2}, nt::DType::Float32);
        input << 1.0f, 2.0f,
                 3.0f, 4.0f;

        nt::Tensor check({1, 1, 1, 1}, nt::DType::Float32);
        check << 4.0f;

        auto out = nt::adaptive_max_pool2d(input = input, output_size = {1, 1});
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check), "Value mismatch");
    });

    run_test("max_unpool1d - basic", [] {
        nt::Tensor input({1, 1, 2}, nt::DType::Float32);
        input << 3.0f, 4.0f;

        nt::Tensor indices({1, 1, 2}, nt::DType::int64);
        indices << 1, 3;

        nt::Tensor check({1, 1, 4}, nt::DType::Float32);
        check << 0.0f, 3.0f, 0.0f, 4.0f;

        auto out = nt::max_unpool1d(input = input, indices = indices, kernel_size = 2, stride = 2);
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check), "Value mismatch");
    });

    run_test("fractional_max_pool2d - basic", [] {
        nt::Tensor input({1, 1, 4, 4}, nt::DType::Float32);
        input << 1, 2, 3, 4,
                 5, 6, 7, 8,
                 9,10,11,12,
                 13,14,15,16;

        nt::Tensor out = nt::fractional_max_pool2d(input = input, kernel_size = {2, 2}, output_size = {2, 2});
        nt::utils::throw_exception(out.shape() == nt::Tensor({1, 1, 2, 2}).shape(), "Shape mismatch");
    });

    run_test("max_unpool2d - basic", [] {
        nt::Tensor input({1, 1, 2, 2}, nt::DType::Float32);
        input << 6.0f, 8.0f,
                 10.0f, 16.0f;

        nt::Tensor indices({1, 1, 2, 2}, nt::DType::int64);
        indices << 5, 7,
                   10, 15;

        nt::Tensor check({1, 1, 4, 4}, nt::DType::Float32);
        check.fill_(0);
        check(0, 0, 1, 1) = 6.0f;
        check(0, 0, 1, 3) = 8.0f;
        check(0, 0, 2, 2) = 10.0f;
        check(0, 0, 3, 3) = 16.0f;

        auto out = nt::max_unpool2d(input = input, indices = indices, kernel_size = {2, 2}, stride = {2, 2});
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check), "Value mismatch");
    });

    run_test("max_unpool3d - basic", [] {
        nt::Tensor input({1, 1, 1, 1, 2}, nt::DType::Float32);
        input << 5.0f, 6.0f;

        nt::Tensor indices({1, 1, 1, 1, 2}, nt::DType::int64);
        indices << 3, 7;

        nt::Tensor check({1, 1, 2, 2, 4}, nt::DType::Float32);
        check.fill_(0);
        check(0, 0, 0, 0, 3) = 5.0f;
        check(0, 0, 0, 1, 3) = 6.0f;

        auto out = nt::max_unpool3d(input = input, indices = indices, kernel_size = {2, 2, 2}, stride = {2, 2, 2});
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch $ != $", out.shape(), check.shape());
        nt::utils::throw_exception(nt::allclose(out, check), "Value mismatch");
    });

    run_test("adaptive_max_pool3d - basic", [] {
        nt::Tensor input({1, 1, 2, 2, 2}, nt::DType::Float32);
        input << 1, 2,
                 3, 4,
                 5, 6,
                 7, 8;

        nt::Tensor check({1, 1, 1, 1, 1}, nt::DType::Float32);
        check << 8.0f;

        auto out = nt::adaptive_max_pool3d(input = input, output_size = {1, 1, 1});
        nt::utils::throw_exception(out.shape() == check.shape(), "Shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check), "Value mismatch");
    });

    run_test("fractional_max_pool3d - basic", [] {
        nt::Tensor input({1, 1, 4, 4, 4}, nt::DType::Float32);
        for (int i = 0; i < 64; ++i) input[0][0][i / 16][(i / 4) % 4][i % 4] = float(i + 1);

        auto out = nt::fractional_max_pool3d(input = input, kernel_size = {2, 2, 2}, output_size = {2, 2, 2});
        nt::utils::throw_exception(out.shape() == nt::Tensor({1, 1, 2, 2, 2}).shape(), "Shape mismatch");
    });


}

