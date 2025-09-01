#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void matmult_test(){
    using namespace nt::literals;
    // flip test
    run_test("matmult 2d", [] {
        nt::Tensor a = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 2d Ta", [] {
        nt::Tensor a = nt::randn({30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 2d Tb", [] {
        nt::Tensor a = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 2d T(a, b)", [] {
        nt::Tensor a = nt::randn({30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });

    run_test("matmult 3d", [] {
        nt::Tensor a = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d Ta", [] {
        nt::Tensor a = nt::randn({10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d Tb", [] {
        nt::Tensor a = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d T(a, b)", [] {
        nt::Tensor a = nt::randn({10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });

    run_test("matmult 3d bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d Ta bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d Tb bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d T(a, b) bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d Ta bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d Tb bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult 3d T(a, b) bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("matmult tensors (no bigger)", []{
        nt::Tensor a = nt::rand(0, 3, {5, 2, 3, 4}, nt::DType::Float32);
        nt::Tensor b = nt::rand(0, 3, {5, 2, 4, 3}, nt::DType::Float32);
        nt::Tensor a_split = a.split_axis(0);
        nt::Tensor b_split = b.split_axis(0);
        nt::Tensor o = nt::matmult(a_split, b_split).view(5, 2, 3, 3);
        for(int i = 0; i < 5; ++i){
            nt::Tensor a1 = a_split[i].item<nt::Tensor>();
            nt::Tensor b1 = b_split[i].item<nt::Tensor>();
            nt::Tensor o1 = nt::matmult(a1, b1);
            nt::utils::throw_exception(nt::all(o1 == o[i]), "Error tensors did not match at index $", i);
        }
    });
    // run_test("tensor of tensors combine", []{
    //     nt::Tensor a_pre_split = nt::rand(0, 3, {5, 2, 4, 3}, nt::DType::Float32);
    //     nt::Tensor a = a_pre_split.split_axis(0);
    //     int64_t total = a.numel() * (a[0].item<nt::Tensor>().numel() / (a[0].item<nt::Tensor>().shape()[-1] * a[0].item<nt::Tensor>().shape()[-2]));
    //     nt::Tensor o = nt::Tensor::makeNullTensorArray(total);
    //     nt::Tensor* begin = reinterpret_cast<nt::Tensor*>(o.data_ptr());
    //     nt::Tensor* end = reinterpret_cast<nt::Tensor*>(o.data_ptr_end());
    //     nt::Tensor* a_begin = reinterpret_cast<nt::Tensor*>(a.data_ptr());
    //     nt::Tensor* a_end = reinterpret_cast<nt::Tensor*>(a.data_ptr_end());
    //     for(; a_begin != a_end; ++a_begin){
    //         nt::Tensor split = a_begin->split_axis(-3);
    //         nt::Tensor* s_begin = reinterpret_cast<nt::Tensor*>(split.data_ptr());
    //         nt::Tensor* s_end = reinterpret_cast<nt::Tensor*>(split.data_ptr_end());
    //         for(; s_begin != s_end; ++s_begin, ++begin){
    //             *begin = *s_begin;
    //         }
    //     }
    //     std::cout << o << std::endl;
    //     std::cout << a << std::endl;
    // });
    run_test("matmult tensors Ta", []{
        nt::Tensor a = nt::rand(0, 3, {5, 2, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor b = nt::rand(0, 3, {5, 2, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_a = true).view(5, 2, 3, 3);
        for(int i = 0; i < 5; ++i){
            nt::Tensor a1 = a[i].item<nt::Tensor>();
            nt::Tensor b1 = b[i].item<nt::Tensor>();
            nt::Tensor o1 = nt::matmult(input = a1, other = b1, transpose_a = true);
            nt::utils::throw_exception(nt::all(o1 == o[i]), "Error tensors did not match at index $", i);
        }

    });
    run_test("matmult tensors Tb", []{
        nt::Tensor a = nt::randn({5, 2, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor b = nt::randn({5, 2, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true).view(5, 2, 4, 4);
        for(int i = 0; i < 5; ++i){
            nt::Tensor a1 = a[i].item<nt::Tensor>();
            nt::Tensor b1 = b[i].item<nt::Tensor>();
            nt::Tensor o1 = nt::matmult(input = a1, other = b1, transpose_b = true);
            nt::utils::throw_exception(nt::all(o1 == o[i]), "Error tensors did not match at index $", i);
        }

    });

    run_test("matmult tensors Ta (bigger a)", []{
        nt::Tensor a = nt::rand(0, 3, {5, 2, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor b = nt::rand(0, 3, {5, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_a = true).view(5, 2, 3, 3);
        for(int i = 0; i < 5; ++i){
            nt::Tensor a1 = a[i].item<nt::Tensor>();
            nt::Tensor b1 = b[i].item<nt::Tensor>();
            nt::Tensor o1 = nt::matmult(input = a1, other = b1, transpose_a = true);
            nt::utils::throw_exception(nt::all(o1 == o[i]), "Error tensors did not match at index $", i);
        }

    });
    run_test("matmult tensors Tb (bigger a)", []{
        nt::Tensor a = nt::randn({5, 2, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor b = nt::randn({5, 4, 3}, nt::DType::Float32).split_axis(0);
        nt::Tensor o = nt::matmult(input = a, other = b, transpose_b = true).view(5, 2, 4, 4);
        for(int i = 0; i < 5; ++i){
            nt::Tensor a1 = a[i].item<nt::Tensor>();
            nt::Tensor b1 = b[i].item<nt::Tensor>();
            nt::Tensor o1 = nt::matmult(input = a1, other = b1, transpose_b = true);
            nt::utils::throw_exception(nt::all(o1 == o[i]), "Error tensors did not match at index $", i);
        }

    });




    run_test("linear 2d", [] {
        nt::Tensor a = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 2d Ta", [] {
        nt::Tensor a = nt::randn({30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 2d Tb", [] {
        nt::Tensor a = nt::randn({30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 2d T(a, b)", [] {
        nt::Tensor a = nt::randn({30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({14, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });

    run_test("linear 3d", [] {
        nt::Tensor a = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30, 30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d Ta", [] {
        nt::Tensor a = nt::randn({10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({14, 30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d Tb", [] {
        nt::Tensor a = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d T(a, b)", [] {
        nt::Tensor a = nt::randn({10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({14, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({10, 14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });

    run_test("linear 3d bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30, 30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d Ta bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({14, 30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d Tb bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d T(a, b) bigger A", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({14, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30, 30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d Ta bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 30, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({14, 30});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 30}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d Tb bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 30}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({30, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 30, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });
    run_test("linear 3d T(a, b) bigger B", [] {
        nt::Tensor a = nt::randn({5, 10, 30, 14}, nt::DType::Float32);
        nt::Tensor b = nt::randn({10, 14, 30}, nt::DType::Float32);
        nt::Tensor bi = nt::randn({14, 14});
        nt::Tensor o = nt::linear(bias = bi, input = a, weight = b, transpose_b = true, transpose_a = true);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({5, 10, 14, 14}), "Error: Got unexpected shape for ($) ($) = ($)", a.shape(), b.shape(), o.shape());
    });

}

