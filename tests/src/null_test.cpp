#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


void null_test(){
    run_test("Null Tensor ptr is null", [] {
        nt::Tensor a = nt::Tensor::Null();
        float* ptr = reinterpret_cast<float*>(a.data_ptr());
        nt::utils::throw_exception(ptr == nullptr,
                                   "Error, ptr from null tensor was not null");
        const nt::ArrayVoid& arr = a.arr_void();
        nt::utils::throw_exception(arr.is_null(),
                                   "Error, array void from null tensor is not null");
        const float* ptr2 = reinterpret_cast<const float*>(arr.data_ptr());
        nt::utils::throw_exception(ptr2 == nullptr,
                                   "Error, ptr from null tensor null array void is not null");

     });
    run_test("Null ArrayVoid ptr is null", [] {
        nt::ArrayVoid a(nullptr);
        float* ptr = reinterpret_cast<float*>(a.data_ptr());
        nt::utils::throw_exception(ptr == nullptr,
                                   "Error, ptr from null array void was not null");

     });


}

