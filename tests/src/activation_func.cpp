#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>

#define ADD_UNDERSCORE(name) name##_

#define NT_MAKE_RUN_TEST(name)\
    run_test(#name, []{\
        for(const auto& dt : NumberTypes){\
            nt::Tensor x = nt::randn({2, 5, 5});\
            auto y = nt::name(input = x);\
            nt::utils::throw_exception(!nt::any(nt::isnan(y)), "Error: dtype $ after $ had nan values", dt, #name);\
        }\
    });\
    run_test(#name " - self", []{\
        for(const auto& dt : NumberTypes){\
            nt::Tensor x = nt::randn({2, 5, 5});\
            nt::ADD_UNDERSCORE(name)(input = x);\
            nt::utils::throw_exception(!nt::any(nt::isnan(x)), "Error: dtype $ after $ had nan values", dt, #name);\
        }\
    });\


void activation_test(){
    using namespace nt::literals;
    NT_MAKE_RUN_TEST(sigmoid)
    NT_MAKE_RUN_TEST(sqrt)
    NT_MAKE_RUN_TEST(invsqrt)
    NT_MAKE_RUN_TEST(abs)
    NT_MAKE_RUN_TEST(relu)
    NT_MAKE_RUN_TEST(gelu)
    NT_MAKE_RUN_TEST(silu)
    run_test("pow", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor x = nt::randn({2, 3, 10});
            auto y = nt::pow(exponent = 2, input = x);
            x.arr_void().execute_function<nt::WRAP_DTYPES<nt::NumberTypesL> >([](auto begin, auto end){
                for(;begin != end; ++begin){
                    *begin *= *begin;
                }
            });
            nt::utils::throw_exception(!nt::any(nt::isnan(y)), "Error: dtype $ after pow had nan values", dt);
            nt::utils::throw_exception(nt::allclose(x, y), "Error: value mismatch for dtype $", dt);

        }
    });

    run_test("pow (3)", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor x = nt::randn({2, 3, 10});
            auto y = nt::pow(exponent = 3, input = x);
            x.arr_void().execute_function<nt::WRAP_DTYPES<nt::NumberTypesL> >([](auto begin, auto end){
                for(;begin != end; ++begin){
                    *begin *= (*begin * *begin);
                }
            });
            nt::utils::throw_exception(!nt::any(nt::isnan(y)), "Error: dtype $ after pow had nan values", dt);
            nt::utils::throw_exception(nt::allclose(x, y), "Error: value mismatch for dtype $", dt);

        }
    });

    run_test("softplus", [] {
        for(const auto& dt : NumberTypes){
            nt::Tensor x = nt::randn({2, 3, 10});
            auto y = nt::softplus(threshold = 30.0, input = x);
            nt::utils::throw_exception(!nt::any(nt::isnan(y)), "Error: dtype $ after softplus had nan values", dt);
        }
    });

}

#undef ADD_UNDERSCORE


// int main() {
//     conv_tests();
//     return 0;
// }
