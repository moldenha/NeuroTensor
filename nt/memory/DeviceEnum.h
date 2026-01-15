#ifndef NT_DEVICE_ENUM_H__
#define NT_DEVICE_ENUM_H__

namespace nt{

enum class DeviceType : int8_t{
	META = -1, //this is the default device that has no memory associated with it, similar to how C10 does it
	CPU = 0, //the normal CPU device
	CPUShared = 1, //allocated with poisix shm class to make the memory shared acorss multiple cpu's
};

#define NT_GET_DEVICES_FUNC(func, ...)\
    func(cpu, __VA_ARGS__)\


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
