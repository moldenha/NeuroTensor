#include "complex.h"
#include <stdexcept>

namespace nt{
namespace functional{
namespace cpu{

template<typename T>
inline const T* data_ptr_end(const ArrayVoid& in){
    return (const T *)(reinterpret_cast<const uint8_t *>(in.data_ptr()) +
                    (in.Size() * DTypeFuncs::size_of_dtype(in.dtype())));
}

void _to_complex_from_real(const ArrayVoid& in, ArrayVoid& output){
    if(!output.is_contiguous()){
        throw std::invalid_argument("Expected output from nt::functional::cpu::_to_complex_from_real to be contiguous");
    }
    DType dtype = in.dtype();
    if (dtype == DType::Double) {
        complex_128 *start = reinterpret_cast<complex_128 *>(output.data_ptr());
        uint32_t type = in.get_bucket().iterator_type();
        if (type == 1) { // contiguous
            const double *begin = reinterpret_cast<const double *>(in.data_ptr());
            const double *end = data_ptr_end<double>(in);
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        } else if (type == 2) {
            auto begin = in.get_bucket().cbegin_blocked<double>();
            auto end = in.get_bucket().cend_blocked<double>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        } else if (type == 3) {
            auto begin = in.get_bucket().cbegin_list<double>();
            auto end = in.get_bucket().cend_list<double>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }

            return;
        }
        return;
    } else if (dtype == DType::Float) {
        complex_64 *start = reinterpret_cast<complex_64 *>(output.data_ptr());
        uint32_t type = in.get_bucket().iterator_type();
        if (type == 1) {
            const float *begin = reinterpret_cast<const float *>(in.data_ptr());
            const float *end = data_ptr_end<float>(in);
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        } else if (type == 2) {
            auto begin = in.get_bucket().cbegin_blocked<float>();
            auto end = in.get_bucket().cend_blocked<float>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        } else if (type == 3) {
            auto begin = in.get_bucket().cbegin_list<float>();
            auto end = in.get_bucket().cend_list<float>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        }
        return;
    }
    else if (dtype == DType::Float16) {
        complex_32 *start = reinterpret_cast<complex_32 *>(output.data_ptr());
        uint32_t type = in.get_bucket().iterator_type();
        if (type == 1) {
            const float16_t *begin =
                reinterpret_cast<const float16_t *>(in.data_ptr());
            const float16_t *end = data_ptr_end<float16_t>(in);
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        } else if (type == 2) {
            auto begin = in.get_bucket().cbegin_blocked<float16_t>();
            auto end = in.get_bucket().cend_blocked<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        } else if (type == 3) {
            auto begin = in.get_bucket().cbegin_list<float16_t>();
            auto end = in.get_bucket().cend_list<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->real() = *begin;
            }
            return;
        }
        return;
    }
}


void _to_complex_from_imag(const ArrayVoid& in, ArrayVoid& output){
    if(!output.is_contiguous()){
        throw std::invalid_argument("Expected output from nt::functional::cpu::_to_complex_from_imag to be contiguous");
    }
    DType dtype = in.dtype();
    if (dtype == DType::Double) {
        complex_128 *start = reinterpret_cast<complex_128 *>(output.data_ptr());
        uint32_t type = in.get_bucket().iterator_type();
        if (type == 1) { // contiguous
            const double *begin = reinterpret_cast<const double *>(in.data_ptr());
            const double *end = data_ptr_end<double>(in);
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        } else if (type == 2) {
            auto begin = in.get_bucket().cbegin_blocked<double>();
            auto end = in.get_bucket().cend_blocked<double>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        } else if (type == 3) {
            auto begin = in.get_bucket().cbegin_list<double>();
            auto end = in.get_bucket().cend_list<double>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }

            return;
        }
        return;
    } else if (dtype == DType::Float) {
        complex_64 *start = reinterpret_cast<complex_64 *>(output.data_ptr());
        uint32_t type = in.get_bucket().iterator_type();
        if (type == 1) {
            const float *begin = reinterpret_cast<const float *>(in.data_ptr());
            const float *end = data_ptr_end<float>(in);
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        } else if (type == 2) {
            auto begin = in.get_bucket().cbegin_blocked<float>();
            auto end = in.get_bucket().cend_blocked<float>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        } else if (type == 3) {
            auto begin = in.get_bucket().cbegin_list<float>();
            auto end = in.get_bucket().cend_list<float>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        }
        return;
    }
    else if (dtype == DType::Float16) {
        complex_32 *start = reinterpret_cast<complex_32 *>(output.data_ptr());
        uint32_t type = in.get_bucket().iterator_type();
        if (type == 1) {
            const float16_t *begin =
                reinterpret_cast<const float16_t *>(in.data_ptr());
            const float16_t *end = data_ptr_end<float16_t>(in);
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        } else if (type == 2) {
            auto begin = in.get_bucket().cbegin_blocked<float16_t>();
            auto end = in.get_bucket().cend_blocked<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        } else if (type == 3) {
            auto begin = in.get_bucket().cbegin_list<float16_t>();
            auto end = in.get_bucket().cend_list<float16_t>();
            for (; begin != end; ++begin, ++start) {
                start->imag() = *begin;
            }
            return;
        }
        return;
    }

}

}
}
}
