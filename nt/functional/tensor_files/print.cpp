#include "print.h"
#include "../../dtype/ArrayVoid.hpp"

//silence depreciation warnings for certain needed headers
#ifdef _MSC_VER
#ifndef _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING
#endif

#ifndef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif

#endif

namespace nt{
namespace functional{

//this function ensures that int8_t and uint8_t are printed properly, otherwise it just comes out as random-looking characters
//and it is really annoying
template <typename IteratorOut>
void print_byte_tensor_template_func(IteratorOut &data, IteratorOut &end,
                                std::ostream &out, const SizeRef &t_s,
                                bool sub_matrix, uint32_t print_space) {
    if (!sub_matrix) {
        out << t_s << std::endl;
        out << "Tensor(";
        print_space += 7;
    }
    if (t_s.empty()) {
        out << "[]" << std::endl;
        return;
    }
    if (t_s.size() == 1) {
        out << "[";
        for (; (data + 1) != end; ++data)
            out << +(*data) << ",";
        out << +(*data) << "]";
        if (!sub_matrix) {
            if(utils::g_print_dtype_on_tensor){
                out << ", " << t_s << ')' << std::endl;
            }else{
                out <<", " << t_s << ')';
            }
        }
        return;
    }
    if (t_s.size() == 2) {
        const Tensor::size_value_t _rows = t_s.front();
        const Tensor::size_value_t _cols = t_s.back();
        out << "[";
        ++print_space;
        for (Tensor::size_value_t x = 0; x < _rows; ++x) {
            if (x != 0 || !sub_matrix) {
                out << "[";
            }
            for (Tensor::size_value_t y = 0; y < _cols - 1; ++y) {
                out << +(*data) << ",";
                ++data;
            }
            if (x == _rows - 1) {
                out << +(*data) << "]]" << std::endl;
            } else {
                out << +(*data) << "],";
                out << std::endl;
                for (uint32_t i = 0; i < print_space; ++i)
                    out << " ";
            }
            ++data;
        }
        if (!sub_matrix) {
            out << "])" << std::endl;
        } else {
            out << std::endl;
            for (uint32_t i = 0; i < print_space - 1; ++i)
                out << " ";
        }
        return;
    }
    for (Tensor::size_value_t i = 0; i < t_s.front(); ++i) {
        if (!sub_matrix && i != 0) {
            for (Tensor::size_value_t j = 0; j < 7; ++j)
                out << " ";
        }
        out << "[";
        print_byte_tensor_template_func(data, end, out, t_s.pop_front(), true,
                                   print_space + 1);
        if (!sub_matrix) {
            out << "\033[F";
            for (Tensor::size_value_t j = 0; j < t_s.size() - 3; ++j)
                out << "]";
            if (i != t_s.front() - 1) {
                out << ',';
                out << std::endl << std::endl;
            } else {
                out << ")";
                if(utils::g_print_dtype_on_tensor){
                    out << std::endl;
                }
            }
        }
    }
}

template <typename IteratorOut>
void print_tensor_template_func(IteratorOut &data, IteratorOut &end,
                                std::ostream &out, const SizeRef &t_s,
                                bool sub_matrix, uint32_t print_space) {
    
    if (!sub_matrix) {
        out << t_s << std::endl;
        out << "Tensor(";
        print_space += 7;
    }
    if (t_s.empty()) {
        out << "[]" << std::endl;
        return;
    }
    if (t_s.size() == 1) {
        out << "[";
        for (; (data + 1) != end; ++data)
            out << *data << ",";
        out << *data << "]";
        if (!sub_matrix) {
            if(utils::g_print_dtype_on_tensor){
                out << ')' << std::endl;
            }else{
                out << ')';
            }
        }
        return;
    }
    if (t_s.size() == 2) {
        const Tensor::size_value_t _rows = t_s.front();
        const Tensor::size_value_t _cols = t_s.back();
        out << "[";
        ++print_space;
        for (Tensor::size_value_t x = 0; x < _rows; ++x) {
            if (x != 0 || !sub_matrix) {
                out << "[";
            }
            for (Tensor::size_value_t y = 0; y < _cols - 1; ++y) {
                out << *data << ",";
                ++data;
            }
            if (x == _rows - 1) {
                out << *data << "]]" << std::endl;
            } else {
                out << *data << "],";
                out << std::endl;
                for (uint32_t i = 0; i < print_space; ++i)
                    out << " ";
            }
            ++data;
        }
        if (!sub_matrix) {
            out << "])" << std::endl;
        } else {
            out << std::endl;
            for (uint32_t i = 0; i < print_space - 1; ++i)
                out << " ";
        }
        return;
    }
    for (Tensor::size_value_t i = 0; i < t_s.front(); ++i) {
        if (!sub_matrix && i != 0) {
            for (Tensor::size_value_t j = 0; j < 7; ++j)
                out << " ";
        }
        out << "[";
        print_tensor_template_func(data, end, out, t_s.pop_front(), true,
                                   print_space + 1);
        if (!sub_matrix) {
            out << "\033[F";
            for (Tensor::size_value_t j = 0; j < t_s.size() - 3; ++j)
                out << "]";
            if (i != t_s.front() - 1) {
                out << ',';
                out << std::endl << std::endl;
            } else {
                out << ")";
                if(utils::g_print_dtype_on_tensor){
                    out << std::endl;
                }
            }
        }
    }
}

inline static constexpr auto print_tensor_func = [](auto data, auto end,
                                                    std::ostream &out,
                                                    const SizeRef &t_s,
                                                    bool sub_matrix,
                                                    uint32_t print_space) {
    using value_t = utils::IteratorBaseType_t<decltype(data)>;
    if constexpr (std::is_same_v<value_t, int8_t> || std::is_same_v<value_t, uint8_t>){
        print_byte_tensor_template_func<decltype(data)>(
            data, end, out, t_s, false, print_space);
        return;
    }
    if (!sub_matrix) {
        out << "Tensor([";
        print_space += 8;
    }
    if (t_s.empty() || t_s.front() == 0) {
        out << "], {0})";
        return;
    }
    if (t_s.size() == 1) {
        for (; (data + 1) != end; ++data) { // this is the issue, the iterator
                                            // can't handle + 1 for some reason?
            out << *data << ",";
        }
        out << *data << "]";
        if (!sub_matrix) {
            if(utils::g_print_dtype_on_tensor){
                out << ", " << t_s << ')' << std::endl;
            }else{
                out <<", " << t_s << ')';
            }

        }
        return;
    }
    if (t_s.size() == 2) {
        const Tensor::size_value_t _rows = t_s.front();
        const Tensor::size_value_t _cols = t_s.back();
        /* out <<"["; */
        ++print_space;
        for (Tensor::size_value_t x = 0; x < _rows; ++x) {
            if (x != 0 || !sub_matrix) {
                out << "[";
            }
            for (Tensor::size_value_t y = 0; y < _cols - 1; ++y) {
                out << *data << ",";
                ++data;
            }
            if (x == _rows - 1) {
                if (!sub_matrix) {
                    out << *data << "]], " << t_s << ")" << std::endl;
                } else {
                    out << *data << "]]" << std::endl;
                }
            } else {
                out << *data << "],";
                out << std::endl;
                for (uint32_t i = 0; i < print_space; ++i)
                    out << " ";
            }
            ++data;
        }
        /* if(!sub_matrix){out << "], "<<t_s<<")" << std::endl;} */
        if (sub_matrix) {
            out << std::endl;
            for (uint32_t i = 0; i < print_space - 1; ++i)
                out << " ";
        }
        return;
    }
    for (Tensor::size_value_t i = 0; i < t_s.front(); ++i) {
        if (!sub_matrix && i != 0) {
            for (uint32_t j = 0; j < 9; ++j)
                out << " ";
        }
        out << "[";
        print_tensor_template_func<decltype(data)>(
            data, end, out, t_s.pop_front(), true, print_space + 1);
        if (!sub_matrix) {
            out << "\033[F";
            for (Tensor::size_value_t j = 0; j < t_s.size() - 3; ++j)
                out << "]";
            if (i != t_s.front() - 1) {
                /* out << ','; */
                out << std::endl << std::endl;
            } else {
                out << "], " << t_s << ")";
                if(utils::g_print_dtype_on_tensor){
                    out << std::endl;
                }
            }
        }
    }
};


//once there are more devices, add a function to check if a device is compatible with the cpu, 
//if not, then cast it to the cpu and run it again
std::ostream& print(std::ostream &out, const Tensor &_t){
    if (_t.is_null()) {
        return out << "Tensor (Null)";
    }
    if (_t.is_empty()) {
        if(utils::g_print_dtype_on_tensor){
            return out << "Tensor([], " << _t.dtype << ")";
        }else{
            return out << "Tensor([])";
        }
    }
    if (_t.dtype == DType::TensorObj && _t.numel() == 1) {
        return print(out, *reinterpret_cast<const Tensor *>(_t.data_ptr())) << std::endl;
    }
    if (_t.dtype == DType::Bool)
        out << std::boolalpha;
    _t.arr_void().cexecute_function<WRAP_DTYPES<AllTypesL>>(
        print_tensor_func, out, _t.shape(), false, 0);
    if (_t.dtype == DType::Bool)
        out << std::noboolalpha;
    if(utils::g_print_dtype_on_tensor){
        out << std::endl << _t.dtype;
    }
    return out;
}
void print(const Tensor &_t){std::cout << _t << std::endl;}

}
}
