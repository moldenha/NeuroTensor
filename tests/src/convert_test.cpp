#include <nt/nt.h>
#include "test_macros.h"
#include <nt/convert/Convert.h>
#include <nt/dtype/DType_enum.h>
#include <nt/types/Types.h>
// #include <nt/types/Cast16.h>
#include <nt/Tensor.h>
#include <nt/dtype/compatible/DType_compatible.h>

template<typename T>
T init(){
    if constexpr (std::is_same_v<T, ::nt::Tensor>){
        return ::nt::Tensor(::nt::Scalar(1.1));
    }
    else if constexpr(std::is_same_v<T, ::nt::float16_t>){
        float val = 1.1f;
        nt::float16_t out = _NT_FLOAT32_TO_FLOAT16_(val);
        return out;
    }
    else{
        return T(1);
    }
}


template<typename FromList, typename ToList>
int test_convert_impl(){
    static constexpr ::nt::DType FromDType = FromList::next;
    static constexpr ::nt::DType ToDType = ToList::next;
    using from_type = typename nt::DTypeFuncs::dtype_to_type_t<FromDType>;
    using to_type = typename nt::DTypeFuncs::dtype_to_type_t<ToDType>;
    from_type val = init<from_type>();
    to_type c_val = ::nt::convert::convert<ToDType>(val);
    if constexpr (ToList::done){
        if constexpr (FromList::done){
            return 0;
        }
        return test_convert_impl<typename FromList::next_wrapper, ToList>();
    }
    else if constexpr (FromList::done){
        return test_convert_impl<::nt::AllTypesL, typename ToList::next_wrapper>();
    }
    else{
        return test_convert_impl<typename FromList::next_wrapper, ToList>();
    }
}



void convert_test(){
    run_test("convert", [] {
        // float val = 1.1f;
        // nt::float128_t _val = nt::convert::convert<nt::float128_t>(val);
        // std::cout << _val << std::endl;
        test_convert_impl<::nt::AllTypesL, ::nt::AllTypesL>();
    });
}

