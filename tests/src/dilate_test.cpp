#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>

// No multiply dtypes because just moving numbers

void dilate_test(){
    using namespace nt::literals;
    // dilate test
    run_test("dilate1d", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::dilate(a, {1});
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error dilate1d returned only 0");
        nt::Tensor d_ = nt::undilate_(d, {1});
        nt::utils::throw_exception(nt::functional::all(d_ == a), "Error dilation1d filled in incorrectly");
    });
    run_test("dilate2d", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::dilate(a, dilation = {3, 1});
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error dilate2d returned only 0");
        nt::utils::throw_exception(nt::functional::all(nt::undilate_(d, {3, 1}) == a), "Error dilation2d filled in incorrectly");
    });
    run_test("dilate3d", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::dilate(input = a, dilation = {1, 3, 1});
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error dilate3d returned only 0");
        nt::utils::throw_exception(nt::functional::all(nt::undilate_(d, {1, 3, 1}) == a), "Error dilation3d filled in incorrectly");
    });
    run_test("dilate ND", [] {
        {
            nt::Tensor a = nt::arange({2, 5, 5}, nt::DType::Float32);
            nt::Tensor d = nt::dilate(input = a, dilation = {1, 2, 3});
            nt::Tensor d2 = nt::dilate(input = a, dilation = {1, 2, 3}, ntarg_(test) = true);
        }
        {
            nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
            nt::Tensor d = nt::dilate(input = a, dilation = {1, 3, 1}, ntarg_(test) = false);
            nt::Tensor d2 = nt::dilate(input = a, dilation = {1, 3, 1}, ntarg_(test) = true);
            nt::utils::throw_exception(nt::functional::all(d == d2), "Error 3d dilate test does not work for nd dilate");
        }
        {
            nt::Tensor a = nt::randn({30}, nt::DType::Float32);
            nt::Tensor d = nt::dilate(input = a, dilation = {2}, ntarg_(test) = false);
            nt::Tensor d2 = nt::dilate(input = a, dilation = {2}, ntarg_(test) = true);
            nt::utils::throw_exception(nt::functional::all(d == d2), "Error 1d dilate test does not work for nd dilate");
        }
        {
            nt::Tensor a = nt::arange({30, 30}, nt::DType::Float32);
            nt::Tensor d = nt::dilate(input = a, dilation = {3, 1}, ntarg_(test) = false);
            nt::Tensor d2 = nt::dilate(input = a, dilation = {3, 1}, ntarg_(test) = true);
            nt::utils::throw_exception(nt::functional::all(d == d2), "Error 2d dilate test does not work for nd dilate");
        }
        
        {
            nt::Tensor a = nt::randn({5, 10, 10}, nt::DType::Float32);
            nt::Tensor d = nt::dilate(input = a, dilation = {2}, ntarg_(test) = true);
        }

    });

    run_test("undilate1d", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::undilate(a, {1});
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error undilate1d returned only 0");
    });
    run_test("undilate2d", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::undilate(a, dilation = {3, 1});
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error undilate2d returned only 0");
    });
    run_test("undilate3d", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::undilate(dilation = {1, 3, 1}, input = a);
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error undilate3d returned only 0");
    });
    run_test("undilate1d_", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::undilate_(a, {1});
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error undilate1d_ returned only 0");
    });
    run_test("undilate2d_", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::undilate_(a, dilation = {3, 1});
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error undilate2d_ returned only 0");
    });
    run_test("undilate3d_", [] {
        nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
        nt::Tensor d = nt::undilate_(dilation = {1, 3, 1}, input = a);
        nt::utils::throw_exception(!nt::functional::all(d == 0), "Error undilate3d_ returned only 0");
    });
    run_test("undilate_ ND", [] {
        {
            nt::Tensor a = nt::randn({30, 30, 30}, nt::DType::Float32);
            nt::Tensor d = nt::undilate_(input = a, dilation = {1, 3, 1}, ntarg_(test) = false);
            nt::Tensor d2 = nt::undilate_(input = a, dilation = {1, 3, 1}, ntarg_(test) = true);
            nt::utils::throw_exception(nt::functional::all(d == d2), "Error 3d undilate_ test does not work for nd undilate_");
        }
        {
            nt::Tensor a = nt::randn({30, 30}, nt::DType::Float32);
            nt::Tensor d = nt::undilate_(input = a, dilation = {3, 1}, ntarg_(test) = false);
            nt::Tensor d2 = nt::undilate_(input = a, dilation = {3, 1}, ntarg_(test) = true);
            nt::utils::throw_exception(nt::functional::all(d == d2), "Error 2d undilate_ test does not work for nd undilate_");
        }
        {
            nt::Tensor a = nt::randn({30}, nt::DType::Float32);
            nt::Tensor d = nt::undilate_(input = a, dilation = {1}, ntarg_(test) = false);
            nt::Tensor d2 = nt::undilate_(input = a, dilation = {1}, ntarg_(test) = true);
            nt::utils::throw_exception(nt::functional::all(d == d2), "Error 1d undilate_ test does not work for nd undilate_");
        }

    });
}

