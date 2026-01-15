#include <iostream>
#include "macros.h"
#include NT_GET_HEADER_PATH(cpu, add)
// #include NT_GET_HEADER_PATH(mkl, add)


// #define example_macro(str)\
//     make_type(#str, std::make_index_sequence<2>{});



namespace cpu{

void add(const DeviceT& d, int a){
    std::cout << "this is a cpu function add " << a << " and given device is " << d.d << std::endl;
}

void mul(const DeviceT& d, int a){
    std::cout << "this is a cpu function mul " << a << " and given device is " << d.d << std::endl;
}

void overlap(DeviceT& out, const DeviceT& d, float b){
    std::cout << "cpu overlap called, and has out and b is " << b << std::endl;
}

NT_REGISTER_OP(add)
NT_REGISTER_OP(mul)
NT_REGISTER_OP(overlap)

}

namespace mkl{

void add(const DeviceT& d, int a){
    std::cout << "this is a mkl function add " << a << " and given device is " << d.d << std::endl;
}

NT_REGISTER_OP(add)
}


#define HAS_CPU_ADD NT_GET_HEADER_PATH_SELECT_0(cpu, add)


int main(){
    DeviceT d;
    d.d = Device::mkl;
    NT_RUN_BARE_METAL_FUNC(add, d, 10);
    std::cout << "device after is "<<d.d << std::endl;
    NT_RUN_BARE_METAL_FUNC(mul, d, 10);
    std::cout << "device after is "<<d.d << std::endl;
    DeviceT out;
    out.d = Device::mkl;
    NT_RUN_BARE_METAL_FUNC(overlap, out, d, 3.3);
    std::cout << "out device is "<<out.d << std::endl;
    std::cout << "d device is " << d.d << std::endl;
    std::cout << HAS_CPU_ADD << std::endl;
    // static_assert(NT_CHECK_REGISTRY(cpu, add), "Error, add should be registered on the cpu");
    // static_assert(!NT_CHECK_REGISTRY(mkl, add), "Error, add should not be registered on mtl");
    // static_assert(!NT_CHECK_REGISTRY(cpu, mul), "Error, mul should not be registered");
    return 0;
}
