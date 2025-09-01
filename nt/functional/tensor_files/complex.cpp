#include "complex.h"
#include "exceptions.hpp"
#include "fill.h"
#include "../cpu/complex.h"

namespace nt{
namespace functional{

Tensor real(const Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    utils::THROW_EXCEPTION(
        DTypeFuncs::is_complex(t.dtype()),
        "RuntimeError: Expected dtype to be a complex number when running real() but got $", t.dtype());
    
    const DType& dtype = t.dtype();
    ArrayVoid out_vals = t.arr_void().get_bucket().is_strided()
                         ? t.arr_void().copy_strides(true)
                         : t.arr_void().bucket_all_indices(); 
    
    out_vals.get_bucket().dtype = (dtype == DType::Complex128  ? DType::Double
                      : dtype == DType::Complex64 ? DType::Float
                                                  : DType::Float16);
    return Tensor(std::move(out_vals), t.shape()).set_mutability(t.is_mutable());
}
Tensor imag(const Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    utils::THROW_EXCEPTION(
        DTypeFuncs::is_complex(t.dtype()),
        "RuntimeError: Expected dtype to be a complex number when running imag() but got $", t.dtype());
    const DType& dtype = t.dtype();
    ArrayVoid out_vals = t.arr_void().get_bucket().is_strided()
                         ? t.arr_void().copy_strides(true)
                         : t.arr_void().bucket_all_indices(); 
    
    out_vals.get_bucket().dtype = (dtype == DType::Complex128  ? DType::Double
                      : dtype == DType::Complex64 ? DType::Float
                                                  : DType::Float16);
    std::size_t complex_size = DTypeFuncs::size_of_dtype(dtype);
    std::size_t imag_size = DTypeFuncs::size_of_dtype(out_vals.dtype());
    // make sure the types are compatible
    utils::THROW_EXCEPTION((imag_size * 2) == complex_size,
                           "[INTERNAL LOGIC ERROR] Expected to have a halfing value when going from "
                           "complex to imaginary");
    void **begin = out_vals.stride_begin();
    void **end = out_vals.stride_end();
    for (; begin != end; ++begin) {
        *begin = reinterpret_cast<uint8_t *>(*begin) + imag_size;
    }
    return Tensor(std::move(out_vals), t.shape()).set_mutability(t.is_mutable());

}
Tensor to_complex_from_real(const Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    DType dtype = t.dtype();
    utils::THROW_EXCEPTION(
        dtype == DType::Double || dtype == DType::Float ||
            dtype == DType::Float16,
        "RuntimeError: Expected dtype to be a floating number but got $",
        dtype);
    Tensor output = ::nt::functional::zeros(
        t.shape(), (dtype == DType::Double  ? DType::Complex128
                  : dtype == DType::Float ? DType::Complex64
                                          : DType::Complex32));
    cpu::_to_complex_from_real(t.arr_void(), output.arr_void());
    return std::move(output);
}

Tensor to_complex_from_imag(const Tensor& t){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    DType dtype = t.dtype();
    utils::THROW_EXCEPTION(
        dtype == DType::Double || dtype == DType::Float ||
            dtype == DType::Float16,
        "RuntimeError: Expected dtype to be a floating number but got $",
        dtype);
    Tensor output = ::nt::functional::zeros(
        t.shape(), (dtype == DType::Double  ? DType::Complex128
                  : dtype == DType::Float ? DType::Complex64
                                          : DType::Complex32));
    cpu::_to_complex_from_imag(t.arr_void(), output.arr_void());
    return std::move(output);

}

}
}
