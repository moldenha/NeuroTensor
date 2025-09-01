#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void min_max_test(){
    using namespace nt::literals;
    // flip test
    run_test("max (no dim)", [] {
        nt::Tensor a = nt::rand(1, 10, {5, 30, 30}, nt::DType::Float32);
        auto out = nt::max(a);
    });
    run_test("max (dim)", [] {
        nt::Tensor a = nt::rand(1, 10, {5, 3, 8, 4, 2}, nt::DType::Float32);
        auto out = nt::max(a, dim = {1, 4});
        nt::utils::throw_exception(out.values.dims() == (a.dims()-2), "Error: Output values shape is $", out.values.shape());
    });
    run_test("max (dim, keepdim)", [] {
        nt::Tensor a = nt::rand(1, 10, {5, 3, 8, 4, 2}, nt::DType::Float32);
        auto out = nt::max(a, dim = {1, 4}, keepdim = true);
        nt::utils::throw_exception(out.values.dims() == (a.dims()), "Error: Output values shape is $", out.values.shape());
    });
    run_test("min (no dim)", [] {
        nt::Tensor a = nt::rand(1, 10, {5, 30, 30}, nt::DType::Float32);
        auto out = nt::min(a);
    });
    run_test("min (dim)", [] {
        nt::Tensor a = nt::rand(1, 10, {5, 3, 8, 4, 2}, nt::DType::Float32);
        auto out = nt::min(a, dim = {1, 4});
        nt::utils::throw_exception(out.values.dims() == (a.dims()-2), "Error: Output values shape is $", out.values.shape());
    });
    run_test("min (dim, keepdim)", [] {
        nt::Tensor a = nt::rand(1, 10, {5, 3, 8, 4, 2}, nt::DType::Float32);
        auto out = nt::min(a, dim = {1, 4}, keepdim = true);
        nt::utils::throw_exception(out.values.dims() == (a.dims()), "Error: Output values shape is $", out.values.shape());
    });
    
    run_test("maximum", []{
        float val_a = 1.0;
        float val_b = 2.0;
        nt::Tensor a({3, 3}, nt::DType::Float32);
        a << 1.2, 3.3, 2.1,
             5.6, 2.1, 8.9,
             2.1, 1.7, 0.3;

        nt::Tensor b({3, 3}, nt::DType::Float32);
        b << 1.0, 3.4, 2.1,
             3.0, 5.0, 1.9,
             2.3, 4.8, 1.1;

        nt::Tensor expected({3, 3}, nt::DType::Float32);
        expected << 2.0, 3.4, 2.1,
                    5.6, 5.0, 8.9,
                    2.3, 4.8, 2.0;

        auto y = nt::maximum(a, b, val_a, val_b);
        nt::utils::throw_exception(nt::all(y == expected), "Error, got incorrect values $ \n$, \n$", nt::noprintdtype, y.view(-1), expected.view(-1));
    });

    run_test("minimum", []{
        float val_a = 3.0;
        float val_b = 2.0;
        nt::Tensor a({3, 3}, nt::DType::Float32);
        a << 1.2, 3.3, 2.1,
             5.6, 2.1, 8.9,
             2.1, 1.7, 0.3;

        nt::Tensor b({3, 3}, nt::DType::Float32);
        b << 1.0, 3.4, 2.1,
             3.0, 5.0, 1.9,
             2.3, 4.8, 1.1;

        nt::Tensor expected({3, 3}, nt::DType::Float32);
        expected << 1.0, 2.0, 2.0,
                    2.0, 2.0, 1.9,
                    2.0, 1.7, 0.3;

        auto y = nt::minimum(a, b, val_a, val_b);
        nt::utils::throw_exception(nt::all(y == expected), "Error, got incorrect values $ \n$, \n$", nt::noprintdtype, y.view(-1), expected.view(-1));
    });

    run_test("clamp", [] {
        nt::Tensor a = nt::rand(1, 10, {5, 3, 8, 4, 2}, nt::DType::int32);
        nt::Tensor o = nt::clamp(a, min = 2, max = 8);
        nt::Scalar check = nt::max(o).values.toScalar();
        nt::utils::throw_exception(check.to<int64_t>() <= 8, "Error, max after clamp was $", check);
        check = nt::min(o).values.toScalar();
        nt::utils::throw_exception(check.to<int64_t>() <= 2, "Error, max after clamp was $", check);

    });
    //all the above tests are implemented to make sure they simply have working calls
    //below argmin and argmax are more throughly tested because
    //min and max functions rely on argmin and argmax's underlying functions to implement their actual logic
    run_test("argmin", []{
        nt::Tensor a({4, 4}, nt::DType::Float32);
        a << 0.1139,  0.2254, -0.1381,  0.3687,
             1.0100, -1.1975, -0.0102, -0.4732,
            -0.9240,  0.1207, -0.7506, -1.0213,
             1.7809, -1.2960,  0.9384,  0.1438;
        nt::Scalar out_a = nt::argmin(a).toScalar();
        nt::utils::throw_exception(out_a.to<int64_t>() == 13, "Error, expected 13 got $", out_a);
        nt::Tensor out_b = nt::argmin(a, dim = 1);
        nt::Tensor check_b({4}, nt::DType::int64);

        check_b << 2,  1,  3,  1;
        nt::utils::throw_exception(nt::all(check_b == out_b), "Error, got invalid elements for argmin");
        nt::Tensor out_c = nt::argmin(a, dim = 1, keepdim=true);
        nt::utils::throw_exception(nt::all(out_c == check_b.view(4, 1)), "Error, got invalid elements for argmin");
    });
    run_test("argmax", []{
        nt::Tensor a({4, 4}, nt::DType::Float32);
        a << 1.3398,  0.2663, -0.2686,  0.2450,
            -0.7401, -0.8805, -0.3402, -1.1936,
             0.4907, -1.3948, -1.0691, -0.3132,
            -1.6092,  0.5419, -0.2993,  0.3195;
        nt::Scalar out_a = nt::argmax(a).toScalar();
        nt::utils::throw_exception(out_a.to<int64_t>() == 0, "Error, expected 13 got $", out_a);
        nt::Tensor out_b = nt::argmax(a, dim = 1);
        nt::Tensor check_b({4}, nt::DType::int64);
        check_b << 0,  2,  0,  1;
        nt::utils::throw_exception(nt::all(check_b == out_b), "Error, got invalid elements for argmin");
        nt::Tensor out_c = nt::argmax(a, dim = 1, keepdim=true);
        nt::utils::throw_exception(nt::all(out_c == check_b.view(4, 1)), "Error, got invalid elements for argmin");
    });


}

