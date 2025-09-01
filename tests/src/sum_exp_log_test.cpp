#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>
#include <nt/linalg/linalg.h>

void sum_exp_log_test(){
    using namespace nt::literals;
    // sum exp log test
    run_test("log and exp - inverse", [] {
        for(const auto dt : combine(FloatingTypes, ComplexTypes)){
            nt::Tensor a = nt::arange(6, start = nt::complex_64(1), dtype = dt);
            auto exp_log = nt::log(input = nt::exp(input = a));
            nt::utils::throw_exception(nt::allclose(exp_log, a, 1e-5), "Log/exp inverse failed");
        }
    });

    run_test("sum - dim keep", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor a = nt::nums({2, 3}, 2.0f, dtype = dt);
            auto s = nt::sum(a, {1}, keepdim = true);
            nt::Tensor expected = nt::nums({2, 1}, 6.0f, dt);
            nt::utils::throw_exception(nt::allclose(s, expected), "Sum keepdim failed");
        }
    });

    run_test("logsumexp - simple", [] {
        for(const auto dt : combine(FloatingTypes, ComplexTypes)){
            nt::Tensor a = nt::arange(3, dtype = dt).view(1, 3);
            auto out = nt::logsumexp(input = a, dim = {1}, keepdim = true);
            nt::Tensor expected({1, 1}, dt); expected(0) = nt::complex_64(std::log(std::exp(0) + std::exp(1) + std::exp(2)));
            nt::utils::throw_exception(nt::allclose(out, expected, 1e-2), "logsumexp failed for $ $ \n$ \n$", dt, nt::noprintdtype, out.view(-1), expected.view(-1));
        }
    });

}

