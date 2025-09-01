#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void softmax_test(){
    using namespace nt::literals;
    // softmax test
    run_test("softmax", [] {
        for(const auto& dt : combine(FloatingTypes, ComplexTypes)){
            nt::Tensor x({3, 3}, dt);
            x << nt::complex_64(1.124f), nt::complex_64(2.567f), nt::complex_64(4.762f),
                nt::complex_64(-4.135f), nt::complex_64(3.41f), nt::complex_64(-5.887f),
                nt::complex_64(-4.0f),  nt::complex_64(1.23f),  nt::complex_64(0.987);

            auto out = nt::softmax(x);
            
            float summed = out.sum().toScalar().to<float>();
            
            nt::utils::throw_exception(summed > 0.9 && summed < 1.1, "Error, softmax did not normalize to 1 for $", dt);
        }
    });
    run_test("softmax - dim (-1)", [] {
        for(const auto& dt : combine(FloatingTypes, ComplexTypes)){
            nt::Tensor x({3, 3}, dt);
            x << nt::complex_64(1.124f), nt::complex_64(2.567f), nt::complex_64(4.762f),
                nt::complex_64(-4.135f), nt::complex_64(3.41f), nt::complex_64(-5.887f),
                nt::complex_64(-4.0f),  nt::complex_64(1.23f),  nt::complex_64(0.987);

            auto out = nt::softmax(x, dim = -1);
            nt::Tensor summed = nt::sum(out, -1);
            nt::Tensor expected = nt::ones_like(summed);

            nt::utils::throw_exception(nt::all(summed > nt::complex_64(0.9) && summed < nt::complex_64(1.1)),
                                       "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, summed, out.flatten(0, -1));
        }
    });
    run_test("gumbel softmax - dim (-1)", [] {
        for(const auto& dt : combine(FloatingTypes, ComplexTypes)){
            nt::Tensor x({3, 3}, dt);
            x << 1.124f, 2.567f, 4.762f,
                -4.135f, 3.41f, -5.887f,
                -4.0f,  1.23f,  0.987;

            auto out = nt::gumbel_softmax(x, dim = -1, hard = false);
            nt::Tensor summed = nt::sum(out, -1);
            // if(dt != nt::DType::Float16 && dt != nt::DType::Complex32){
            nt::Tensor expected = nt::ones_like(summed);
            // if(nt::any(nt::hasnan(out)) && (dt == nt::DType::Complex32 || dt == nt::DType::Float16))
            //     continue;
            nt::utils::throw_exception(nt::all(summed > nt::complex_64(0.9) && summed < nt::complex_64(1.1)),
                                       "Value mismatch for $ $ \n$ \n$", dt, nt::printdtype, summed, out.flatten(0, -1));
            // }
            // else{
                // if(nt::any(nt::isnan(summed)))
                //     continue;
                // nt::utils::throw_exception(nt::all(summed > nt::complex_64(0.9) && summed < nt::complex_64(1.1)),
                //                             "Value mismatch for $ $ \n$ \n$", dt, nt::noprintdtype, summed, out.flatten(0, -1));
            // }
        }
    });


}
