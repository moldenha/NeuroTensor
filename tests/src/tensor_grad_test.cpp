#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"


// this is an example function to test x = func(x) and make sure it propogates properly
// This syntax is used in models, so it is important to know that it works
// I don't like that it adds an additional "EqualOperator&&Backward" function
// But, there really isn't a way around that
nt::TensorGrad example_forward(nt::TensorGrad x){
    using namespace nt::literals;
    x = x + 10;
    x = nt::relu(x);
    x = x.permute({0, 2, 1});
    x *= 5;
    return nt::sum(x, {-1, -2});
}


void tensor_grad_test(){
    run_test("tensor grad test", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
    });

    run_test("tensor grad plus test (+ scalar)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        auto y = x + 4;
    });
    run_test("tensor grad test 2 (+= scalar)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        x += 4;
    });
    run_test("tensor grad test 3 (+ scalar)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        auto y = x + 4;
        y.backward(nt::rand(0, 4, {3, 4, 5}));
    });
    run_test("tensor grad test 4 (+= scalar, backward)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        x += 4;
        x.backward(nt::rand(0, 4, {3, 4, 5}));
    });
    run_test("tensor grad test 5 (+= grad, backward)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        nt::TensorGrad y(nt::rand(-8, 8, {3, 4, 5}));
        x += y;
        x.backward(nt::rand(0, 4, {3, 4, 5}));
        x.update();
        y.update();
    });

    run_test("tensor grad test 6 (complex)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        nt::TensorGrad y(nt::rand(-8, 8, {3, 4, 5}));
        auto z = x + y;
        // z += nt::rand(0, 10, {5});
        z -= nt::rand(0, 10, {5});
        auto m = z / nt::TensorGrad(nt::rand(3, 4, {3, 4, 5}));
        m *= 20;
        auto o = nt::relu(m);
        o += x;
        o.backward(nt::rand(0, 4, {3, 4, 5}));
        
    });
    run_test("tensor grad test 7 func(x = func(x))", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        nt::TensorGrad y(nt::rand(-8, 8, {3, 4, 5}));
        auto z = x + y;
        auto logits = example_forward(z);
        logits.backward(nt::rand(0, 3, logits.shape()));
        x.update();
        y.update();
    });
    run_test("tensor grad test 8 view change operator (+=)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        nt::TensorGrad y(nt::rand(-8, 8, {4, 5}));
        auto o = x[1] + y;
        o.backward(nt::rand(0, 3, o.shape()));
        const nt::Tensor& grad = x.grad();
        nt::utils::throw_exception(nt::all(grad[0] == 0), "Error, expected gradient[0] to be all 0");
        nt::utils::throw_exception(nt::all(grad[2] == 0), "Error, expected gradient[2] to be all 0");
        nt::utils::throw_exception(nt::any(grad[1] != 0), "Error, expected gradient[1] to not be all 0");

    });

    run_test("tensor grad test 9 view change operator (fill)", []{
        nt::TensorGrad x(nt::rand(0, 3, {3, 4, 5}));
        nt::TensorGrad y(nt::rand(-8, 8, {3, 4, 5}));
        // nt::fill_(x[1], 0);
        auto a = x[1];
        a += 10;
        auto b = a[1];
        b *= 2000;
        b[1 <nt::range> 3] = 0;
        x[2] *= 1000;
        auto o = x + y;
        o.backward(nt::rand(0, 3, o.shape()));
        nt::Tensor& grad = x.grad();
        nt::utils::throw_exception(nt::sum(grad[2]).item<float>() > 1000, "Error, gradient [2] sum is expected to be greater than 1000");
        nt::utils::throw_exception(nt::all(grad[1][1][1 <nt::range> 3] == 0), "Error gradient [1][1][ 1 <nt::range> 3 ] is expected to be 0");
        nt::utils::throw_exception(nt::sum(grad[1][1]).item<float>() > 1000, "Error, expected gradient[1][1] sum to be greater than 1000");

    });



}
