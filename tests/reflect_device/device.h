#ifndef NT_DEVICE_H__
#define NT_DEVICE_H__
#include <iostream>

enum class Device{
    meta,
    cpu, 
    mkl, 
    cuda
};

std::ostream& operator<<(std::ostream& os, const Device& d){
    switch(d){
        case Device::meta:
            return os << "Device::meta";
        case Device::cpu:
            return os << "Device::cpu";
        case Device::mkl:
            return os << "Device::mkl";
        case Device::cuda:
            return os << "Device::cuda";
    }
}

struct DeviceT{
    Device d;
    DeviceT to(Device d_){return DeviceT{d_};}
    DeviceT& to_(Device d_){d = d_; return *this;}
};


#endif
