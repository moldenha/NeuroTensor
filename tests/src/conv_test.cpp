#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/mp/simde_traits.h>
#include <nt/convert/Convert.h>


// nt::mp::SimdTraits_avx2<nt::float16_t>::Type load_masked(const nt::float16_t* data, const nt::mp::mask_type_avx2& mask){
//     int16_t mask_vals[16];
//     nt::mp::SimdTraits_avx2<int16_t>::storeu((nt::mp::SimdTraits_avx2<int16_t>::Type*)mask_vals, mask);
//     nt::float16_t n_data[8];
//     for(int i = 0; i < 8; ++i){
//         n_data[i] = mask_vals[i] == 0 ? 0 : data[i];
//     }
//     return nt::mp::SimdTraits_avx2<nt::float16_t>::loadu(n_data);

// }

// nt::mp::SimdTraits_avx<int16_t>::Type load_masked(const int16_t* data, const nt::mp::mask_type_avx& mask){
//     int16_t mask_vals[nt::mp::SimdTraits_avx<int16_t>::pack_size];
//     nt::mp::SimdTraits_avx<int16_t>::storeu((nt::mp::SimdTraits_avx<int16_t>::Type*)mask_vals, mask);
//     int16_t n_data[nt::mp::SimdTraits_avx<int16_t>::pack_size];
//     for(int i = 0; i < nt::mp::SimdTraits_avx<int16_t>::pack_size; ++i){
//         n_data[i] = mask_vals[i] == 0 ? 0 : data[i];
//     }
//     return nt::mp::SimdTraits_avx<int16_t>::loadu(n_data);

// }


// #define NT_MAKE_AVX_TEST_(type)\
//     run_test(#type " load masked avx", []{\
//         if constexpr (nt::mp::simde_supported_avx<type>::value){\
//         for(int i = 1; i < nt::mp::SimdTraits_avx<type>::pack_size; ++i){\
//             type* arr = new type[i];\
//             for(int j = 0; j < i; ++j)\
//              arr[j] = nt::convert::convert<type>(float(3.0));\
//             for(int j = i; j < nt::mp::SimdTraits_avx<type>::pack_size; ++j)\
//              arr[j] = nt::convert::convert<type>(float(-3.0));\
//             nt::mp::mask_type_avx mask = nt::mp::generate_mask_avx<type>(i);\
//             nt::mp::simde_type_avx<type> loaded = nt::mp::SimdTraits_avx<type>::load_masked(arr, mask);\
//         }\
//         }\
//     });\

// #define NT_MAKE_AVX2_TEST_(type)\
//     run_test(#type " load masked avx", []{\
//         using arr_to_t = std::conditional_t<std::is_integer_v<type>, std::make_signed_t<type>, type>;\
//         if constexpr (nt::mp::simde_supported_avx2<type>::value){\
//         for(int i = 1; i < nt::mp::SimdTraits_avx2<type>::pack_size; ++i){\
//             type* arr = new type[i];\
//             for(int j = 0; j < i; ++j)\
//              arr[j] = nt::convert::convert<type>(float(3.0));\
//             for(int j = i; j < nt::mp::SimdTraits_avx2<type>::pack_size; ++j)\
//              arr[j] = nt::convert::convert<type>(float(-3.0));\
//             nt::mp::mask_type_avx2 mask = nt::mp::generate_mask_avx2<type>(i);\
//             nt::mp::simde_type_avx2<type> loaded = nt::mp::SimdTraits_avx2<type>::load_masked((arr_to_t*)arr, mask);\
//         }\
//         }\
//     });



template<typename T>
struct my_make_signed{
    using type = T;
};

template<>
struct my_make_signed<uint64_t>{
    using type = int64_t;
};

template<>
struct my_make_signed<uint32_t>{
    using type = int32_t;
};


template<>
struct my_make_signed<uint16_t>{
    using type = int16_t;
};


template<>
struct my_make_signed<uint8_t>{
    using type = int8_t;
};


#ifdef SIMDE_ARCH_X86_AVX2
template<typename T>
void make_avx2_test(std::string name){
    run_test(name.c_str(), []{
        using arr_to_t = typename my_make_signed<T>::type;
        if constexpr (nt::mp::simde_supported_avx2<T>::value){
            for(int i = 1; i < nt::mp::SimdTraits_avx2<T>::pack_size; ++i){
                T* arr = new T[i];
                for(int j = 0; j < i; ++j)
                 arr[j] = nt::convert::convert<T>(float(3.0));
                // for(int j = i; j < nt::mp::SimdTraits_avx2<T>::pack_size; ++j)
                //  arr[j] = nt::convert::convert<T>(float(-3.0));
                nt::mp::mask_type_avx2 mask = nt::mp::generate_mask_avx2<T>(i);
                nt::mp::simde_type_avx2<T> loaded = nt::mp::SimdTraits_avx2<T>::load_masked((arr_to_t*)arr, mask);
            }
        }
    });
}

#else 
// just something to fill the space
template<typename T> void make_avx2_test(std::string name){
    run_test(name.c_str(), []{
        using arr_to_t = typename my_make_signed<T>::type;
    });
}

#endif

template<typename T>
void make_avx_test(std::string name){
    run_test(name.c_str(), []{
        using arr_to_t = typename my_make_signed<T>::type;
        if constexpr (nt::mp::simde_supported_avx<T>::value){
            for(int i = 1; i < nt::mp::SimdTraits_avx<T>::pack_size; ++i){
                T* arr = new T[i];
                for(int j = 0; j < i; ++j)
                 arr[j] = nt::convert::convert<T>(float(3.0));
                // for(int j = i; j < nt::mp::SimdTraits_avx<T>::pack_size; ++j)
                //  arr[j] = nt::convert::convert<T>(float(-3.0));
                nt::mp::mask_type_avx mask = nt::mp::generate_mask_avx<T>(i);
                nt::mp::simde_type_avx<T> loaded = nt::mp::SimdTraits_avx<T>::load_masked((const arr_to_t*)arr, mask);
            }
        }
    });
}

#define NT_MAKE_AVX_TEST_(type) make_avx_test<type>(#type " load masked avx");
#define NT_MAKE_AVX2_TEST_(type) make_avx2_test<type>(#type " load masked avx2");



void conv_test(){
    using namespace nt::literals;
    NT_MAKE_AVX_TEST_(nt::float16_t);
    NT_MAKE_AVX_TEST_(float);
    NT_MAKE_AVX_TEST_(double);
    NT_MAKE_AVX_TEST_(nt::complex_64);
    NT_MAKE_AVX_TEST_(nt::complex_128);
    NT_MAKE_AVX_TEST_(nt::complex_32);
    NT_MAKE_AVX_TEST_(int64_t);
    // NT_MAKE_AVX_TEST_(uint64_t);
    NT_MAKE_AVX_TEST_(int32_t);
    NT_MAKE_AVX_TEST_(uint32_t);
    NT_MAKE_AVX_TEST_(int16_t);
    NT_MAKE_AVX_TEST_(uint16_t);
    NT_MAKE_AVX_TEST_(int8_t);
    NT_MAKE_AVX_TEST_(uint8_t);
    NT_MAKE_AVX2_TEST_(nt::float16_t);
    NT_MAKE_AVX2_TEST_(float);
    NT_MAKE_AVX2_TEST_(double);
    NT_MAKE_AVX2_TEST_(nt::complex_32);
    NT_MAKE_AVX2_TEST_(nt::complex_64);
    NT_MAKE_AVX2_TEST_(nt::complex_128);
    NT_MAKE_AVX2_TEST_(int64_t);
    // NT_MAKE_AVX2_TEST_(uint64_t);
    NT_MAKE_AVX2_TEST_(int32_t);
    NT_MAKE_AVX2_TEST_(uint32_t);
    NT_MAKE_AVX2_TEST_(int16_t);
    NT_MAKE_AVX2_TEST_(uint16_t);
    NT_MAKE_AVX2_TEST_(int8_t);
    NT_MAKE_AVX2_TEST_(uint8_t);

    // conv1d test
    run_test("conv1d", [] {
        for (const auto& dt : FloatingTypes){
            nt::Tensor x = nt::randn({1, 3, 10}, dtype = dt);
            nt::Tensor w = nt::randn({2, 3, 3}, dtype = dt);
auto y = nt::conv1d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv1d call for $", dt);
        }
        for (const auto& dt : ComplexTypes){
            nt::Tensor x = nt::randn({1, 3, 10}, dtype = dt);
            nt::Tensor w = nt::randn({2, 3, 3}, dtype = dt);
            auto y = nt::conv1d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv1d call for $", dt);
        }
    });

    // conv2d test
    run_test("conv2d", [] {
        nt::Tensor _x = nt::randn({1, 3, 16, 16}, dtype = nt::DType::int32);
        nt::Tensor _w = nt::randn({4, 3, 3, 3}, dtype = nt::DType::int32);
        auto _y = nt::conv2d(_x, _w, stride = 1, padding = 1);
        nt::utils::throw_exception(!_x.is_null(), "Error x was made null after conv2d call for $", nt::DType::int32);
        nt::Tensor __x = nt::randn({1, 3, 16, 16}, dtype = nt::DType::int16);
        nt::Tensor __w = nt::randn({4, 3, 3, 3}, dtype = nt::DType::int16);
        auto __y = nt::conv2d(__x, __w, stride = 1, padding = 1);
        nt::utils::throw_exception(!__x.is_null(), "Error x was made null after conv2d call for $", nt::DType::int32);
        for(const auto& dt : FloatingTypes){
            nt::Tensor x = nt::randn({1, 3, 16, 16}, dtype = dt);
            nt::Tensor w = nt::randn({4, 3, 3, 3}, dtype = dt);
            auto y = nt::conv2d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x = nt::randn({1, 3, 16, 16}, dtype = dt);
            nt::Tensor w = nt::randn({4, 3, 3, 3}, dtype = dt);
            auto y = nt::conv2d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
     });

    // conv3d test
    run_test("conv3d", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x = nt::randn({1, 3, 8, 8, 8}, dtype = dt);
            nt::Tensor w = nt::randn({4, 3, 3, 3, 3}, dtype = dt);
            auto y = nt::conv3d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x = nt::randn({1, 3, 8, 8, 8}, dtype = dt);
            nt::Tensor w = nt::randn({4, 3, 3, 3, 3}, dtype = dt);
            auto y = nt::conv3d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
     });

    // conv_transpose1d test
    run_test("conv_transpose1d", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x = nt::randn({1, 3, 10}, dtype = dt);
            nt::Tensor w = nt::randn({3, 2, 3}, dtype = dt);
            auto y = nt::conv_transpose1d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x = nt::randn({1, 3, 10}, dtype = dt);
            nt::Tensor w = nt::randn({3, 2, 3}, dtype = dt);
            auto y = nt::conv_transpose1d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
     });

    // conv_transpose2d test
    run_test("conv_transpose2d", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x = nt::randn({1, 3, 16, 16}, dtype = dt);
            nt::Tensor w = nt::randn({3, 4, 3, 3}, dtype = dt);
            auto y = nt::conv_transpose2d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x = nt::randn({1, 3, 16, 16}, dtype = dt);
            nt::Tensor w = nt::randn({3, 4, 3, 3}, dtype = dt);
            auto y = nt::conv_transpose2d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
    });

    // conv_transpose3d test
    run_test("conv_transpose3d", [] {
        for(const auto& dt : FloatingTypes){
            nt::Tensor x = nt::randn({1, 3, 8, 8, 8}, dtype = dt);
            nt::Tensor w = nt::randn({3, 4, 3, 3, 3}, dtype = dt);
            auto y = nt::conv_transpose3d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
        for(const auto& dt : ComplexTypes){
            nt::Tensor x = nt::randn({1, 3, 8, 8, 8}, dtype = dt);
            nt::Tensor w = nt::randn({3, 4, 3, 3, 3}, dtype = dt);
            auto y = nt::conv_transpose3d(x, w, stride = 1, padding = 1);
            nt::utils::throw_exception(!x.is_null(), "Error x was made null after conv2d call for $", dt);
        }
    });

}



#undef NT_MAKE_AVX_TEST_ 
#undef NT_MAKE_AVX2_TEST_ 


// int main() {
//     conv_tests();
//     return 0;
// }
