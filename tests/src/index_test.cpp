#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void index_test(){
    using namespace nt::literals;
    // index test
    run_test("at", [] {
        nt::Tensor a = nt::randn({3, 30, 30}, nt::DType::Float32);
        nt::Tensor o = nt::at(input = a, idx = 0);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({30, 30}), "Error, got shape $ after at call", o.shape());
    });
    run_test("at_tensor_split", [] {
        nt::Tensor a = nt::rand(0, 10, {3, 20, 20}, nt::DType::Float32);
        nt::Tensor maxed = nt::functional::max_pool2d(/*Tensor input = */ a, /*utils::my_tuple kernel_size = */ 2,
                  /*utils::my_tuple stride =*/ -1, /*utils::my_tuple padding =*/ 0,
                  /*utils::my_tuple dilation =*/ 1, /*bool ceil_mode =*/ false,
                  /*bool return_indices =*/ true, /*bool get_bools = */false);
        auto [values, indices] = nt::get<2>(maxed);
        nt::Tensor setting = nt::at_tensor_split(a.flatten(-1, -2), splitting = -2, idx = indices.flatten(-1, -2));
    });
    run_test("at_tensor_split self_set", [] {
        nt::Tensor a = nt::rand(0, 10, {3, 20, 20}, nt::DType::Float32);
        nt::Tensor maxed = nt::functional::max_pool2d(/*Tensor input = */ a, /*utils::my_tuple kernel_size = */ 2,
                  /*utils::my_tuple stride =*/ -1, /*utils::my_tuple padding =*/ 0,
                  /*utils::my_tuple dilation =*/ 1, /*bool ceil_mode =*/ false,
                  /*bool return_indices =*/ true, /*bool get_bools = */false);
        auto [values, indices] = nt::get<2>(maxed);
        nt::Tensor out({3, 100}, a.dtype());
        nt::at_tensor_split(a.flatten(-1, -2), splitting = -2, idx = indices.flatten(-1, -2), output = out);
        nt::utils::throw_exception(out.dims() != 0, "at tensor split did not set out");
    });
    run_test("index_select", [] {
        nt::Tensor a = nt::rand(0, 10, {3, 20, 3}, nt::DType::Float32);
        nt::Tensor o = nt::index_select(idx = nt::functional::vector_to_tensor(std::vector<int64_t>({5, 2, 1})),
                                        input = a, dim = -2);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({3, 3, 3}), "Error, got wrong shape for idx");
     });
    run_test("index_except", [] {
        nt::Tensor a = nt::rand(0, 10, {3, 20, 20}, nt::DType::Float32);
        nt::Tensor o = nt::index_except(idx = 5, input = a, dim = -2);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({3, 19, 20}), "Error, got wrong shape for idx");
     });
     run_test("select", []{
        nt::Tensor a = nt::rand(0, 10, {3, 20, 20}, nt::DType::Float32);
        nt::Tensor o = nt::select(idx = 5, input = a, dim = -2);
        nt::utils::throw_exception(o.shape() == nt::SizeRef({3, 1, 20}), "Error, got wrong shape for idx");
    });
    run_test("variadic tensor at", []{
        nt::Tensor a = nt::rand(0, 10, {3, 4, 5, 4, 8}, nt::DType::Float32);
        nt::Tensor o = a(1, 2, 1);
        nt::utils::throw_exception(nt::all(o == a[1][2][1]), "Error, indexing does not work, $", o.shape());
    });
    run_test("variadic tensor ranges at", []{
        nt::Tensor a = nt::rand(0, 10, {3, 4, 5, 4, 8}, nt::DType::Float32);
        nt::Tensor o = a(1, 2 <nt::range> -1, 1 <nt::range> -2, 2);
        nt::utils::throw_exception(nt::all(o == a[{nt::range_(1), nt::range_(2, -1), nt::range_(1, -2), nt::range_(2)}]), "Error, indexing does not work");
        nt::utils::throw_exception(o.shape() == nt::SizeRef({1, 2, 3, 1, 8}), "Error: got unexpected shape $", o.shape());
    });
 




}

