#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


#define NT_MAKE_OPERATOR_TEST(name, op)\
    run_test(#name " scalar", [] {\
        for(const auto& dt : NumberTypes){\
            nt::Tensor a = nt::rand(-3, 3, {5}, dt);\
            nt::Tensor out = nt::name(a, 4);\
            nt::Tensor check({5}, dt);\
            a.arr_void().execute_function<nt::WRAP_DTYPES<nt::NumberTypesL>>([](auto begin, auto end, auto begin2){\
                nt::Scalar num(4);\
                using value_t = nt::utils::IteratorBaseType_t<decltype(begin)>;\
                value_t official_num = num.to<value_t>();\
                for(;begin != end; ++begin, ++begin2){*begin2 = *begin op official_num;}\
            }, check.arr_void());\
            nt::utils::throw_exception(nt::allclose(out, check), "Error, operation $ was not close for $ $ \n$ \n$ \n$ \n$", #op, dt, nt::noprintdtype, a.view(-1), check.view(-1), out.view(-1), nt::isclose(out, check));\
        }\
    });\
    run_test(#name " tensors", []{\
        for(const auto& dt : NumberTypes){\
            nt::Tensor a = nt::rand(-3, 3, {4, 1}, dt);\
            nt::Tensor b = nt::rand(-2, 5, {1, 4}, dt);\
            if(std::string(#name) == "divide"){\
                b[b == 0] = 1;\
                a[a == 0] = 1;\
            }\
            nt::Tensor out = nt::name(a, b);\
            nt::utils::throw_exception(out.shape() == nt::SizeRef({4,4}), "Error got unexpected shape $", out.shape());\
            nt::Tensor check({4, 4}, dt);\
            a.arr_void().execute_function<nt::WRAP_DTYPES<nt::NumberTypesL>>([&check](auto begin, auto end, auto begin2){\
                    using value_t = nt::utils::IteratorBaseType_t<decltype(begin)>;\
                    value_t* o_begin = reinterpret_cast<value_t*>(check.data_ptr());\
                    value_t* o_end = reinterpret_cast<value_t*>(check.data_ptr_end());\
                    for(int i = 0; i < 4; ++i){\
                        for(int j = 0; j < 4; ++j, ++o_begin){\
                            *o_begin = begin[i] op begin2[j];\
                        }\
                    }\
            }, b.arr_void());\
            nt::utils::throw_exception(nt::allclose(out, check), "Error, operation was not close");\
        }\
    });


void operator_test(){
    using namespace nt::literals;
    NT_MAKE_OPERATOR_TEST(multiply, *);
    NT_MAKE_OPERATOR_TEST(add, +);
    NT_MAKE_OPERATOR_TEST(subtract, -);
    NT_MAKE_OPERATOR_TEST(divide, /);
    run_test("remainder - tensor % tensor (3x3)", [] {
        nt::Tensor a({3, 3}, nt::DType::Float32);
        nt::Tensor b({3, 3}, nt::DType::Float32);
        a << 5.5f, -3.2f, 9.9f,
             -4.5f, 7.1f, -2.0f,
             8.3f, -6.4f, 3.3f;

        b << 2.0f, 2.0f, 4.0f,
             3.0f, -2.0f, -3.0f,
             5.0f, 3.0f, -2.0f;

        nt::Tensor check({3, 3}, nt::DType::Float32);
        check << 1.5f, 0.8f, 1.9f,
                 1.5f, -0.9f, -2.0f,
                 3.3f, 2.6f, -0.7f;

        nt::Tensor out = nt::remainder(input = a, other = b);
        nt::utils::throw_exception(out.shape() == check.shape(), "Error, modulo output shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Error, modulo output mismatch");
    });    
    run_test("remainder - tensor % scalar (3x3)", [] {
        nt::Tensor a({3, 3}, nt::DType::Float32);
        a << 7.1f, -3.5f, 4.4f,
             -8.3f, 9.7f, -5.2f,
             9.3f, 8.1f, 1.0f;

        float scalar = 3.0f;

        nt::Tensor check({3, 3}, nt::DType::Float32);
        check << 1.1f, 2.5f, 1.4f,
                 0.7f, 0.7f, 0.8f,
                 0.3f, 2.1f, 1.0f;

        nt::Tensor out = nt::remainder(input = a, other = scalar);  
        nt::utils::throw_exception(out.shape() == check.shape(), "Error, modulo output shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Error, modulo scalar output mismatch");
    });

    run_test("fmod - tensor % tensor (3x3)", [] {
        nt::Tensor a({3, 3}, nt::DType::Float32);
        nt::Tensor b({3, 3}, nt::DType::Float32);
        a << 5.5f, -3.2f, 9.9f,
             -4.5f, 7.1f, -2.0f,
             8.3f, -6.4f, 3.3f;

        b << 2.0f, 2.0f, 4.0f,
             3.0f, -2.0f, -3.0f,
             5.0f, 3.0f, -2.0f;

        nt::Tensor check({3, 3}, nt::DType::Float32);
        check << 1.5f ,-1.2f ,1.9f,
                -1.5f ,1.1f ,-2.0f,
                 3.3f ,-0.4f ,1.3f;

        nt::Tensor out = nt::fmod(input = a, other = b);
        nt::utils::throw_exception(out.shape() == check.shape(), "Error, fmod output shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Error, fmod output mismatch");
    });

    run_test("fmod - tensor % scalar (3x3)", [] {
        nt::Tensor a({3, 3}, nt::DType::Float32);
        a << 7.1f, -3.5f, 4.4f,
             -8.3f, 9.7f, -5.2f,
             9.3f, 8.1f, 1.0f;

        float scalar = 3.0f;

        nt::Tensor check({3, 3}, nt::DType::Float32);
        check << 1.1f, -0.5f, 1.4f,
                 -2.3f, 0.7f, -2.2f,
                  0.3f, 2.1f, 1.0f;

        nt::Tensor out = nt::fmod(input = a, other = scalar);  
        nt::utils::throw_exception(out.shape() == check.shape(), "Error, fmod scalar output shape mismatch");
        nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Error, fmod scalar output mismatch");
    });

    run_test("inverse - reciprocal", [] {
        for(const auto& dt : combine(FloatingTypes, ComplexTypes)){
            nt::Tensor a({2, 3}, dt);
            a << 2.0f, -4.0f, 0.5f,
                 -1.0f, 10.0f, -0.2f;

            nt::Tensor out = nt::inverse(a);  // inverse(x) = 1/x
            nt::utils::throw_exception(nt::DTypeFuncs::is_complex(out.dtype()) || nt::DTypeFuncs::is_floating(out.dtype()),
                                       "Error, for input dtype $ to inverse, got an output dtype of $ that is neither complex or floating", dt, out.dtype());
            nt::Tensor check({2, 3}, out.dtype());
            check << 0.5f, -0.25f, 2.0f,
                     -1.0f, 0.1f, -5.0f;

            nt::utils::throw_exception(out.shape() == check.shape(), "Error, inverse output shape mismatch");
            nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Error, inverse output mismatch $ $ \n$ \n$ \n$", dt, nt::printdtype, a.view(-1), check.view(-1), out.view(-1));
        }
        for(const auto& dt : IntegerTypes){
            nt::Tensor a({2, 3}, dt);
            a << 2, 4, 1,
                 1, 10, 1;

            nt::Tensor out = nt::inverse(a);  // inverse(x) = 1/x
            nt::utils::throw_exception(nt::DTypeFuncs::is_complex(out.dtype()) || nt::DTypeFuncs::is_floating(out.dtype()),
                                       "Error, for input dtype $ to inverse, got an output dtype of $ that is neither complex or floating", dt, out.dtype());
            nt::Tensor check({2, 3}, out.dtype());
            check << 0.5f, 0.25f, 1,
                     1, 0.1, 1;

            nt::utils::throw_exception(out.shape() == check.shape(), "Error, inverse output shape mismatch");
            nt::utils::throw_exception(nt::allclose(out, check, 1e-5), "Error, inverse output mismatch $ $ \n$ \n$ \n$", dt, nt::printdtype, a.view(-1), check.view(-1), out.view(-1));
        }

    });
}


#undef NT_MAKE_OPERATOR_TEST 
