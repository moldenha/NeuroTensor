#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void complex_test(){
    using namespace nt::literals;
    // cat test
    run_test("real", [] {
        nt::Tensor a = nt::randn({3, 300, 300}, nt::DType::Complex64);
        nt::complex_64 val_before = *reinterpret_cast<nt::complex_64*>(a.data_ptr());
        float before_real = val_before.real();
        float before_imag = val_before.imag();
        nt::Tensor r = nt::real(a);
        r += 10;
        nt::complex_64 val_after = *reinterpret_cast<nt::complex_64*>(a.data_ptr());
        float after_real = val_after.real();
        float after_imag = val_after.imag();
        nt::utils::throw_exception((before_real + 10) == after_real, "Error expected after real to be 10 greater than before real but got $ and $", before_real, after_real);
        nt::utils::throw_exception(before_imag == after_imag, "Error expected after imag to be equal to before imag but got $ and $", before_imag, after_imag);
    });
    run_test("imag", [] {
        nt::Tensor a = nt::randn({3, 300, 300}, nt::DType::Complex64);
        nt::complex_64 val_before = *reinterpret_cast<nt::complex_64*>(a.data_ptr());
        float before_real = val_before.real();
        float before_imag = val_before.imag();
        nt::Tensor r = nt::imag(a);
        r += 10;
        nt::complex_64 val_after = *reinterpret_cast<nt::complex_64*>(a.data_ptr());
        float after_real = val_after.real();
        float after_imag = val_after.imag();
        nt::utils::throw_exception((before_imag + 10) == after_imag, "Error expected after imag to be 10 greater than before imag but got $ and $", before_imag, after_imag);
        nt::utils::throw_exception(before_real == after_real, "Error expected after real to be equal to before real but got $ and $", before_real, after_real);
    });

    run_test("to_complex_from_real", [] {
        nt::Tensor a = nt::randn({3, 300, 300}, nt::DType::Float32);
        nt::Tensor r = nt::to_complex_from_real(input = a);
        nt::utils::throw_exception(r.dtype() == nt::DType::Complex64, "Expected float32 to make complex of 64 instead got dtye $", r.dtype());
        nt::complex_64 val_after = *reinterpret_cast<nt::complex_64*>(r.data_ptr());
        float after_real = val_after.real();
        float after_imag = val_after.imag();
        nt::utils::throw_exception(after_real != 0, "Error expected after real to be not 0 but got  $", after_real);
        nt::utils::throw_exception(after_imag == 0, "Error expected after imag to be 0 but got $", after_imag);
    });
    run_test("to_complex_from_imag", [] {
        nt::Tensor a = nt::randn({3, 300, 300}, nt::DType::Float32);
        nt::Tensor r = nt::to_complex_from_imag(input = a);
        nt::utils::throw_exception(r.dtype() == nt::DType::Complex64, "Expected float32 to make complex of 64 instead got dtye $", r.dtype());
        nt::complex_64 val_after = *reinterpret_cast<nt::complex_64*>(r.data_ptr());
        float after_real = val_after.real();
        float after_imag = val_after.imag();
        nt::utils::throw_exception(after_real == 0, "Error expected after real to be 0 but got  $", after_real);
        nt::utils::throw_exception(after_imag != 0, "Error expected after imag to be not 0 but got $", after_imag);
    });

}

