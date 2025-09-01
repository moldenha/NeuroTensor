#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void round_test(){
    using namespace nt::literals;
    // round test
    run_test("round", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x({3, 3}, dt);
            x << 1.124f, 2.567f, 4.762f,
                -4.135f, 3.41f, -5.887f,
                -4.0f,   1.23f,  0.987;

            nt::Tensor expected({3,3}, dt);
            expected << 1.0f, 3.0f, 5.0f,
                       -4.0f, 3.0f,-6.0f,
                       -4.0f, 1.0f, 1.0;

            auto out = nt::round(x);

            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x({3, 3}, dt);
            x << nt::complex_64(1.124f), nt::complex_64(2.567f), nt::complex_64(4.762f),
                nt::complex_64(-4.135f), nt::complex_64(3.41f), nt::complex_64(-5.887f),
                nt::complex_64(-4.0f),   nt::complex_64(1.23f),  nt::complex_64(0.987);

            nt::Tensor expected({3,3}, dt);
            expected << nt::complex_64(1.0f), nt::complex_64(3.0f), nt::complex_64(5.0f),
                       nt::complex_64(-4.0f), nt::complex_64(3.0f), nt::complex_64(-6.0f),
                       nt::complex_64(-4.0f), nt::complex_64(1.0f), nt::complex_64(1.0);

            auto out = nt::round(x);

            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }

    });
    run_test("trunc", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x({3, 3}, dt);
            x << 1.124f, 2.567f, 4.762f,
                -4.135f, 3.41f, -5.887f,
                -4.0f,   1.23f,  0.987;

            nt::Tensor expected({3,3}, dt);
            expected << 1.0f, 2.0f, 4.0f,
                       -4.0f, 3.0f,-5.0f,
                       -4.0f, 1.0f, 0.0;

            auto out = nt::trunc(x);
            
            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x({3, 3}, dt);
            x << nt::complex_64(1.124f), nt::complex_64(2.567f), nt::complex_64(4.762f),
                nt::complex_64(-4.135f), nt::complex_64(3.41f), nt::complex_64(-5.887f),
                nt::complex_64(-4.0f),   nt::complex_64(1.23f),  nt::complex_64(0.987);

            nt::Tensor expected({3,3}, dt);
            expected << nt::complex_64(1.0f), nt::complex_64(2.0f), nt::complex_64(4.0f),
                       nt::complex_64(-4.0f), nt::complex_64(3.0f), nt::complex_64(-5.0f),
                       nt::complex_64(-4.0f), nt::complex_64(1.0f), nt::complex_64(0.0);

            auto out = nt::trunc(x);

            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }

    });
    run_test("floor", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x({3, 3}, dt);
            x << 1.124f, 2.567f, 4.762f,
                -4.135f, 3.41f, -5.887f,
                -4.0f,   1.23f,  0.987;

            nt::Tensor expected({3,3}, dt);
            expected << 1.0f, 2.0f, 4.0f,
                       -5.0f, 3.0f,-6.0f,
                       -4.0f, 1.0f, 0.0;

            auto out = nt::floor(x);

            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }
        
        for(const auto& dt : ComplexTypes){
            nt::Tensor x({3, 3}, dt);
            x << nt::complex_64(1.124f), nt::complex_64(2.567f), nt::complex_64(4.762f),
                nt::complex_64(-4.135f), nt::complex_64(3.41f), nt::complex_64(-5.887f),
                nt::complex_64(-4.0f),   nt::complex_64(1.23f),  nt::complex_64(0.987);

            nt::Tensor expected({3,3}, dt);
            expected << nt::complex_64(1.0f), nt::complex_64(2.0f), nt::complex_64(4.0f),
                       nt::complex_64(-5.0f), nt::complex_64(3.0f), nt::complex_64(-6.0f),
                       nt::complex_64(-4.0f), nt::complex_64(1.0f), nt::complex_64(0.0);

            auto out = nt::floor(x);

            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }
    });
    run_test("ceil", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x({3, 3}, dt);
            x << 1.124f, 2.567f, 4.762f,
                -4.135f, 3.41f, -5.887f,
                -4.0f,   1.23f,  0.987;

            nt::Tensor expected({3,3}, dt);
            expected << 2.0f, 3.0f, 5.0f,
                       -4.0f, 4.0f,-5.0f,
                       -4.0f, 2.0f, 1.0;

            auto out = nt::ceil(x);

            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x({3, 3}, dt);
            x << nt::complex_64(1.124f), nt::complex_64(2.567f), nt::complex_64(4.762f),
                nt::complex_64(-4.135f), nt::complex_64(3.41f), nt::complex_64(-5.887f),
                nt::complex_64(-4.0f),   nt::complex_64(1.23f),  nt::complex_64(0.987);

            nt::Tensor expected({3,3}, dt);
            expected << nt::complex_64(2.0f), nt::complex_64(3.0f), nt::complex_64(5.0f),
                       nt::complex_64(-4.0f), nt::complex_64(4.0f), nt::complex_64(-5.0f),
                       nt::complex_64(-4.0f), nt::complex_64(2.0f), nt::complex_64(1.0);

            auto out = nt::ceil(x);

            nt::utils::throw_exception(out.shape() == expected.shape(), "Shape mismatch for $", dt);
            nt::utils::throw_exception(nt::allclose(out, expected), "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }
    });


}
