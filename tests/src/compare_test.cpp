#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <vector>
#include <sstream>
#include <string>


#include <nt/nt.h>
#include "test_macros.h"
#include <nt/mp/simde_traits.h>
#include <type_traits>
#include <iostream>


template<typename T, std::enable_if_t<!std::is_same_v<T, nt::float16_t> && !std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>, bool> = true>
inline std::string vec_to_string(const std::vector<T>& vec){
    std::ostringstream os;
    if constexpr (std::is_same_v<T, bool>){ os << std::boolalpha; }
    os << '{';
    for(uint32_t i = 0; i < vec.size()-1; ++i){
        os << vec[i] << ", ";
    }
    if constexpr (std::is_same_v<T, bool>){ os << vec.back() << '}' << std::noboolalpha; }
    else{ os << vec.back() << '}'; }
    return os.str();
}

template<typename T, std::enable_if_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, bool> = true>
inline std::string vec_to_string(const std::vector<T>& vec){
    std::ostringstream os;
    os << '{';
    for(uint32_t i = 0; i < vec.size()-1; ++i){
        os << int(vec[i]) << ", ";
    }
    os <<  int(vec.back()) << '}';
    return os.str();
}

template<typename T, std::enable_if_t<std::is_same_v<T, nt::float16_t>, bool> = true>
inline std::string vec_to_string(const std::vector<T>& vec){
    std::ostringstream os;
    os << '{';
    for(uint32_t i = 0; i < vec.size()-1; ++i){
        os << _NT_FLOAT16_TO_FLOAT32_(vec[i]) << ", ";
    }
    os <<  _NT_FLOAT16_TO_FLOAT32_(vec.back()) << '}';
    return os.str();
}



template<typename T>
inline nt::mp::simde_type<T> load_l(T* l){
    if constexpr (std::is_integral_v<T> || std::is_unsigned_v<T>){
        return nt::mp::SimdTraits<T>::loadu(reinterpret_cast<const nt::mp::simde_type<T>*>(l));
    }else{
        return nt::mp::SimdTraits<T>::loadu(l);
    }
}

#define NT_SIMD_EQUAL_TEST(type)\
    run_test("SIMD equal " #type, []{\
        static_assert(nt::mp::simde_supported_v<type>, "Error type" #type " unsupported frr simd pre-defined traits");\
        type vals_a[nt::mp::SimdTraits<type>::pack_size];\
        type vals_b[nt::mp::SimdTraits<type>::pack_size];\
        type start = type(10);\
        for(int i = 0; i < nt::mp::SimdTraits<type>::pack_size; ++i){\
            if(i % 2 == 0){\
                vals_a[i] = start;\
                vals_b[i] = start;\
            }\
            else{\
                vals_a[i] = start - 2;\
                vals_b[i] = start + 3;\
            }\
        }\
        nt::mp::simde_type<type> a = load_l(vals_a);\
        nt::mp::simde_type<type> b = load_l(vals_b);\
        bool comp_vals[nt::mp::SimdTraits<type>::pack_size];\
        nt::mp::SimdTraits<type>::store_compare_equal(a, b, comp_vals);\
        std::vector<type> a_vec(vals_a, &vals_a[nt::mp::SimdTraits<type>::pack_size]);\
        std::vector<type> b_vec(vals_b, &vals_b[nt::mp::SimdTraits<type>::pack_size]);\
        std::vector<bool> c_vec(comp_vals, &comp_vals[nt::mp::SimdTraits<type>::pack_size]);\
        for(int i = 0; i < nt::mp::SimdTraits<type>::pack_size; ++i){\
            nt::utils::throw_exception(c_vec[i] == (i % 2 == 0), "Error, got wrong comparison\nComp vals are: $\nVals a: $\nVals b: $", vec_to_string(c_vec), vec_to_string(a_vec), vec_to_string(b_vec));\
        }\
    });\
    run_test("SIMD not equal " #type, []{\
        static_assert(nt::mp::simde_supported_v<type>, "Error type" #type " unsupported frr simd pre-defined traits");\
        type vals_a[nt::mp::SimdTraits<type>::pack_size];\
        type vals_b[nt::mp::SimdTraits<type>::pack_size];\
        type start = type(10);\
        for(int i = 0; i < nt::mp::SimdTraits<type>::pack_size; ++i){\
            if(i % 2 == 0){\
                vals_a[i] = start;\
                vals_b[i] = start;\
            }\
            else{\
                vals_a[i] = start - 2;\
                vals_b[i] = start + 3;\
            }\
        }\
        nt::mp::simde_type<type> a = load_l(vals_a);\
        nt::mp::simde_type<type> b = load_l(vals_b);\
        bool comp_vals[nt::mp::SimdTraits<type>::pack_size];\
        nt::mp::SimdTraits<type>::store_compare_not_equal(a, b, comp_vals);\
        std::vector<type> a_vec(vals_a, &vals_a[nt::mp::SimdTraits<type>::pack_size]);\
        std::vector<type> b_vec(vals_b, &vals_b[nt::mp::SimdTraits<type>::pack_size]);\
        std::vector<bool> c_vec(comp_vals, &comp_vals[nt::mp::SimdTraits<type>::pack_size]);\
        for(int i = 0; i < nt::mp::SimdTraits<type>::pack_size; ++i){\
            nt::utils::throw_exception(c_vec[i] == (i % 2 != 0), "Error, got wrong comparison\nComp vals are: $\nVals a: $\nVals b: $", vec_to_string(c_vec), vec_to_string(a_vec), vec_to_string(b_vec));\
        }\
    });\
    run_test("SIMD less than equal " #type, []{\
        static_assert(nt::mp::simde_supported_v<type>, "Error type" #type " unsupported frr simd pre-defined traits");\
        type vals_a[nt::mp::SimdTraits<type>::pack_size];\
        type vals_b[nt::mp::SimdTraits<type>::pack_size];\
        type start = type(10);\
        for(int i = 0; i < nt::mp::SimdTraits<type>::pack_size; ++i){\
            if(i % 2 == 0){\
                vals_a[i] = start;\
                vals_b[i] = start + 10;\
            }\
            else{\
                vals_a[i] = start + 5;\
                vals_b[i] = start - 3;\
            }\
        }\
        nt::mp::simde_type<type> a = load_l(vals_a);\
        nt::mp::simde_type<type> b = load_l(vals_b);\
        bool comp_vals[nt::mp::SimdTraits<type>::pack_size];\
        nt::mp::SimdTraits<type>::store_compare_less_than_equal(a, b, comp_vals);\
        std::vector<type> a_vec(vals_a, &vals_a[nt::mp::SimdTraits<type>::pack_size]);\
        std::vector<type> b_vec(vals_b, &vals_b[nt::mp::SimdTraits<type>::pack_size]);\
        std::vector<bool> c_vec(comp_vals, &comp_vals[nt::mp::SimdTraits<type>::pack_size]);\
        for(int i = 0; i < nt::mp::SimdTraits<type>::pack_size; ++i){\
            nt::utils::throw_exception(c_vec[i] == (i % 2 == 0), "Error, got wrong comparison\nComp vals are: $\nVals a: $\nVals b: $", vec_to_string(c_vec), vec_to_string(a_vec), vec_to_string(b_vec));\
        }\
    });\


#define NT_MAKE_COMPARE_TEST(name)\
    run_test(#name, []{\
        nt::Tensor a = nt::rand(4, 10, {1, 5}, nt::DType::Float32);\
        nt::Tensor b = nt::rand(3, 8,  {1, 5}, nt::DType::Float32);\
        auto y = nt::name(a, b);\
        int64_t amt = nt::amount_of(input = y, val = true);\
    });

void compare_test(){
    using namespace nt::literals;
    NT_MAKE_COMPARE_TEST(equal)
    NT_MAKE_COMPARE_TEST(not_equal)
    NT_MAKE_COMPARE_TEST(less_than)
    NT_MAKE_COMPARE_TEST(greater_than)
    NT_MAKE_COMPARE_TEST(less_than_equal)
    NT_MAKE_COMPARE_TEST(greater_than_equal)
    NT_SIMD_EQUAL_TEST(int8_t)
    NT_SIMD_EQUAL_TEST(uint8_t)
    NT_SIMD_EQUAL_TEST(int16_t)
    NT_SIMD_EQUAL_TEST(uint16_t)
    NT_SIMD_EQUAL_TEST(int32_t)
    NT_SIMD_EQUAL_TEST(uint32_t)
    NT_SIMD_EQUAL_TEST(int64_t)
    NT_SIMD_EQUAL_TEST(nt::float16_t)
    NT_SIMD_EQUAL_TEST(float)
    NT_SIMD_EQUAL_TEST(double)
    NT_SIMD_EQUAL_TEST(nt::complex_32)
    NT_SIMD_EQUAL_TEST(nt::complex_64)
    NT_SIMD_EQUAL_TEST(nt::complex_128)

}

#undef NT_MAKE_COMPARE_TEST 
#undef NT_MAKE_COMPARE_TEST 

