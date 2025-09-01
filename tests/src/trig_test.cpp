#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>
#include <cmath>
#include <nt/mp/simde_traits.h>

#define ADD_UNDERSCORE(name) _##name
#define ADD_UNDERSCORE2(name) name##_

namespace std{

// #define NT_MAKE_TRIG_STD_FUNC(name)\
// nt::complex_32 name(nt::complex_32 a){\
//     return nt::complex_32(_NT_FLOAT32_TO_FLOAT16_(name(_NT_FLOAT16_TO_FLOAT32_(a.real()))), _NT_FLOAT32_TO_FLOAT16_(name(_NT_FLOAT16_TO_FLOAT32_(a.imag()))));\
// }\
// nt::complex_64 name(nt::complex_64 a){\
//     return nt::complex_64(name(a.real()), name(a.imag()));
// }\
// nt::complex_128 name(nt::complex_128 a){\
//     double r__ = a.real();\
//     double r = name(r__);\
//     double i = name(a.imag());\
//     return nt::complex_128(r, i);
// }\

inline nt::complex_64 atanh(nt::complex_64 a){
    return nt::complex_64(atanh(a.real()), atanh(a.imag()));
}
inline nt::complex_64 asinh(nt::complex_64 a){
    return nt::complex_64(asinh(a.real()), asinh(a.imag()));
}
inline nt::complex_64 acosh(nt::complex_64 a){
    return nt::complex_64(acosh(a.real()), acosh(a.imag()));
}

inline nt::complex_32 atanh(nt::complex_32 a){
    return nt::complex_32(_NT_FLOAT16_TO_FLOAT32_(atanh(_NT_FLOAT16_TO_FLOAT32_(a.real()))), _NT_FLOAT32_TO_FLOAT16_(atanh(_NT_FLOAT16_TO_FLOAT32_(a.imag()))));
}
inline nt::complex_32 asinh(nt::complex_32 a){
    return nt::complex_32(_NT_FLOAT16_TO_FLOAT32_(asinh(_NT_FLOAT16_TO_FLOAT32_(a.real()))), _NT_FLOAT32_TO_FLOAT16_(asinh(_NT_FLOAT16_TO_FLOAT32_(a.imag()))));
}
inline nt::complex_32 acosh(nt::complex_32 a){
    return nt::complex_32(_NT_FLOAT16_TO_FLOAT32_(acosh(_NT_FLOAT16_TO_FLOAT32_(a.real()))), _NT_FLOAT32_TO_FLOAT16_(acosh(_NT_FLOAT16_TO_FLOAT32_(a.imag()))));
}

// NT_MAKE_TRIG_STD_FUNC(asinh)
// // NT_MAKE_TRIG_STD_FUNC(atanh)

// #undef NT_MAKE_TRIG_STD_FUNC 
    
}


#define NT_MAKE_INV_INLINE_FUNC_(operation, name)\
template<typename T>\
inline T _nt_##name(T element) noexcept {return T(1)/operation(element);}

// NOTE: The nt::float16_t variable has a lower precision
//       and SIMD traits handle it slightly differently
//       this is why the atol and rtol are raised specifically for float16 variables

#define NT_MAKE_TRIG_TEST_(name, other, op)\
    run_test(#name, []{\
        for(const auto& dt : FloatingTypes){\
            if(dt == nt::DType::Float16) continue;\
            nt::Tensor a({3, 3}, dt);\
            a << 3.3f, 4.6f, 4.1f,\
                -1.2f, 4.1f, 0.9f,\
                 0.1f, -3.0f, 10.5f;\
            auto b = nt::name(a);\
            nt::Tensor expected({3, 3}, dt);\
            expected << op(3.3f), op(4.6f), op(4.1f),\
                        op(-1.2f), op(4.1f), op(0.9f),\
                         op(0.1f), op(-3.0f), op(10.5f);\
            nt::utils::throw_exception(nt::allclose(b, expected, equal_nan = true), "Error " #name " failed to produce correct results $ $\n $\n $", \
                nt::noprintdtype, b.flatten(0, -1), expected.flatten(0, -1), dt);\
        }\
        for(const auto& dt : ComplexTypes){\
            if(dt == nt::DType::Complex32) continue;\
            nt::Tensor a({3, 3}, dt);\
            a << nt::complex_64(3.3f), nt::complex_64(4.6f), nt::complex_64(4.1f),\
                nt::complex_64(-1.2f), nt::complex_64(4.1f), nt::complex_64(0.9f),\
                 nt::complex_64(0.1f), nt::complex_64(-3.0f), nt::complex_64(10.5f);\
            auto b = nt::name(a);\
            nt::Tensor expected({3, 3}, dt);\
            expected << op(nt::complex_64(3.3f)), op(nt::complex_64(4.6f)), op(nt::complex_64(4.1f)),\
                        op(nt::complex_64(-1.2f)), op(nt::complex_64(4.1f)), op(nt::complex_64(0.9f)),\
                         op(nt::complex_64(0.1f)), op(nt::complex_64(-3.0f)), op(nt::complex_64(10.5f));\
            nt::utils::throw_exception(nt::allclose(b, expected, equal_nan = true, ::nt::literals::atol = 1e-1, rtol = 1e-1), "Error " #name " failed to produce correct results $ \n$\n $\n \n$ \n$", \
                nt::noprintdtype, b.flatten(0, -1), expected.flatten(0, -1), nt::isclose(b, expected, equal_nan = true, ::nt::literals::atol = 1e-1, rtol = 1e-1), dt);\
        }\
        nt::DType dt = nt::DType::Float16;\
        nt::Tensor a({3, 3}, dt);\
        a << 3.3f, 4.6f, 4.1f,\
            -1.2f, 4.1f, 0.9f,\
             0.1f, -3.0f, 10.5f;\
        auto b = nt::name(a);\
        nt::Tensor expected({3, 3}, dt);\
        expected << nt::float16_t(op(3.3f)), nt::float16_t(op(4.6f)), nt::float16_t(op(4.1f)),\
                    nt::float16_t(op(-1.2f)), nt::float16_t(op(4.1f)), nt::float16_t(op(0.9f)),\
                     nt::float16_t(op(0.1f)), nt::float16_t(op(-3.0f)), nt::float16_t(op(10.5f));\
        nt::utils::throw_exception(nt::allclose(b, expected, equal_nan = true, ::nt::literals::atol = 1e-1, rtol = 1e-1),\
            "Error " #name " failed to produce correct results $ $\n $\n $", \
            nt::noprintdtype, b.flatten(0, -1), expected.flatten(0, -1), dt);\
        dt = nt::DType::Complex32;\
        nt::Tensor a2({3, 3}, dt);\
        a2 << nt::complex_32(3.3f), nt::complex_32(4.6f), nt::complex_32(4.1f),\
             nt::complex_32(-1.2f), nt::complex_32(4.1f), nt::complex_32(0.9f),\
             nt::complex_32(0.1f), nt::complex_32(-3.0f), nt::complex_32(10.5f);\
        auto b2 = nt::name(a2);\
        nt::Tensor expected2({3, 3}, dt);\
        expected2 << op(nt::complex_32(3.3f)), op(nt::complex_32(4.6f)), op(nt::complex_32(4.1f)),\
                    op(nt::complex_32(-1.2f)), op(nt::complex_32(4.1f)), op(nt::complex_32(0.9f)),\
                    op(nt::complex_32(0.1f)), op(nt::complex_32(-3.0f)), op(nt::complex_32(10.5f));\
        nt::utils::throw_exception(nt::allclose(b2, expected2, equal_nan = true, ::nt::literals::atol = 1e-1, rtol = 1e-1),\
            "Error " #name " failed to produce correct results $ $\n $\n $", \
            nt::noprintdtype, b2.flatten(0, -1), expected2.flatten(0, -1), dt);\
    });\
    run_test(#name " - self", []{\
       for(const auto& dt : FloatingTypes){\
            if(dt == nt::DType::Float16) continue;\
            nt::Tensor a({3, 3}, dt);\
            a << 3.3f, 4.6f, 4.1f,\
                -1.2f, 4.1f, 0.9f,\
                 0.1f, -3.0f, 10.5f;\
            nt::ADD_UNDERSCORE2(name)(a);\
            nt::Tensor expected({3, 3}, dt);\
            expected << op(3.3f), op(4.6f), op(4.1f),\
                        op(-1.2f), op(4.1f), op(0.9f),\
                         op(0.1f), op(-3.0f), op(10.5f);\
            nt::utils::throw_exception(nt::allclose(a, expected, equal_nan = true), "Error " #name " failed to produce correct results $ $\n $\n $", \
                nt::noprintdtype, a.flatten(0, -1), expected.flatten(0, -1), dt);\
        } \
       for(const auto& dt : ComplexTypes){\
            if(dt == nt::DType::Complex32) continue;\
            nt::Tensor a({3, 3}, dt);\
            a << nt::complex_64(3.3f), nt::complex_64(4.6f), nt::complex_64(4.1f),\
                 nt::complex_64(-1.2f), nt::complex_64(4.1f), nt::complex_64(0.9f),\
                 nt::complex_64(0.1f), nt::complex_64(-3.0f), nt::complex_64(10.5f);\
            nt::ADD_UNDERSCORE2(name)(a);\
            nt::Tensor expected({3, 3}, dt);\
            expected << op(nt::complex_64(3.3f)), op(nt::complex_64(4.6f)), op(nt::complex_64(4.1f)),\
                        op(nt::complex_64(-1.2f)), op(nt::complex_64(4.1f)), op(nt::complex_64(0.9f)),\
                         op(nt::complex_64(0.1f)), op(nt::complex_64(-3.0f)), op(nt::complex_64(10.5f));\
            nt::utils::throw_exception(nt::allclose(a, expected, equal_nan = true, ::nt::literals::atol = 1e-1, rtol = 1e-1), "Error " #name " failed to produce correct results $ $\n $\n $", \
                nt::noprintdtype, a.flatten(0, -1), expected.flatten(0, -1), dt);\
        } \
        nt::DType dt = nt::DType::Float16;\
        nt::Tensor a({3, 3}, dt);\
        a << 3.3f, 4.6f, 4.1f,\
            -1.2f, 4.1f, 0.9f,\
             0.1f, -3.0f, 10.5f;\
        nt::ADD_UNDERSCORE2(name)(a);\
        nt::Tensor expected({3, 3}, dt);\
        expected << nt::float16_t(op(3.3f)), nt::float16_t(op(4.6f)), nt::float16_t(op(4.1f)),\
                    nt::float16_t(op(-1.2f)), nt::float16_t(op(4.1f)), nt::float16_t(op(0.9f)),\
                     nt::float16_t(op(0.1f)), nt::float16_t(op(-3.0f)), nt::float16_t(op(10.5f));\
        nt::utils::throw_exception(nt::allclose(a, expected, equal_nan = true, ::nt::literals::atol = 1e-1, rtol = 1e-1),\
            "Error " #name " failed to produce correct results $ $\n $\n $", \
            nt::noprintdtype, a.flatten(0, -1), expected.flatten(0, -1), dt);\
        dt = nt::DType::Complex32;\
        nt::Tensor a2({3, 3}, dt);\
        a2 << nt::complex_32(3.3f), nt::complex_32(4.6f), nt::complex_32(4.1f),\
             nt::complex_32(-1.2f), nt::complex_32(4.1f), nt::complex_32(0.9f),\
             nt::complex_32(0.1f), nt::complex_32(-3.0f), nt::complex_32(10.5f);\
        nt::ADD_UNDERSCORE2(name)(a2);\
        nt::Tensor expected2({3, 3}, dt);\
        expected2 << op(nt::complex_32(3.3f)), op(nt::complex_32(4.6f)), op(nt::complex_32(4.1f)),\
                    op(nt::complex_32(-1.2f)), op(nt::complex_32(4.1f)), op(nt::complex_32(0.9f)),\
                    op(nt::complex_32(0.1f)), op(nt::complex_32(-3.0f)), op(nt::complex_32(10.5f));\
        nt::utils::throw_exception(nt::allclose(a2, expected2, equal_nan = true, ::nt::literals::atol = 1e-1, rtol = 1e-1),\
            "Error " #name " failed to produce correct results $ $\n $\n $", \
            nt::noprintdtype, a2.flatten(0, -1), expected2.flatten(0, -1), dt);\
    });


NT_MAKE_INV_INLINE_FUNC_(std::tanh, cotanh)
NT_MAKE_INV_INLINE_FUNC_(std::tan, cotan)
NT_MAKE_INV_INLINE_FUNC_(std::sinh, csch)
NT_MAKE_INV_INLINE_FUNC_(std::sin, csc)
NT_MAKE_INV_INLINE_FUNC_(std::cosh, sech)
NT_MAKE_INV_INLINE_FUNC_(std::cos, sec)

void trig_test(){
    using namespace nt::literals;

    NT_MAKE_TRIG_TEST_(tanh, tanh, std::tanh);
    NT_MAKE_TRIG_TEST_(tan, tan, std::tan);
    NT_MAKE_TRIG_TEST_(atan, atan, std::atan);
    NT_MAKE_TRIG_TEST_(atanh, atanh, std::atanh);
    NT_MAKE_TRIG_TEST_(cotanh, cotanh, _nt_cotanh);
    NT_MAKE_TRIG_TEST_(cotan, cotan, _nt_cotanh);

    NT_MAKE_TRIG_TEST_(sinh, sinh, std::sinh);
    NT_MAKE_TRIG_TEST_(sin, sin, std::sin);
    NT_MAKE_TRIG_TEST_(asin, asin, std::asin);
    NT_MAKE_TRIG_TEST_(asinh, asinh, std::asinh);
    NT_MAKE_TRIG_TEST_(csch, csch, _nt_csch);
    NT_MAKE_TRIG_TEST_(csc, csc, _nt_csc);

    NT_MAKE_TRIG_TEST_(cosh, cosh, std::cosh);
    NT_MAKE_TRIG_TEST_(cos, cos, std::cos);
    NT_MAKE_TRIG_TEST_(acos, acos, std::acos);
    NT_MAKE_TRIG_TEST_(acosh, acosh, std::acosh);
    NT_MAKE_TRIG_TEST_(sech, sech, _nt_sech);
    NT_MAKE_TRIG_TEST_(sec, sec, _nt_sec);

}

