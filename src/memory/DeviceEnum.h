#ifndef _NT_DEVICE_ENUM_H_
#define _NT_DEVICE_ENUM_H_

namespace nt{

enum class DeviceType : int8_t{
	META = -1, //this is the default device that has no memory associated with it, similar to how C10 does it
	CPU = 0, //the normal CPU device
	CPUShared = 2, //allocated with poisix shm class to make the memory shared acorss multiple cpu's
};

constexpr DeviceType dCPU = DeviceType::CPU;
constexpr DeviceType dCPUShared = DeviceType::CPUShared;
constexpr DeviceType dMETA = DeviceType::META;

inline std::ostream& operator << (std::ostream& os, const DeviceType& dt) noexcept{
	switch(dt){
		case DeviceType::CPU:
			return os << "DeviceType::CPU";
		case DeviceType::CPUShared:
			return os << "DeviceType::CPUShared";
		default:
			return os << "UnknownDevice";
	}
}


}

#endif
