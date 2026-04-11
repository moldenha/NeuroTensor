#ifndef NT_DEVICE_ENUM_H__
#define NT_DEVICE_ENUM_H__

namespace nt{

enum class DeviceType : int8_t{
	META = -1, //this is the default device that has no memory associated with it, similar to how C10 does it
	CPU = 0, //the normal CPU device
    MTL = 1
};


enum class MemoryLayout : int8_t {
    None = -1, // this is for the META device
    Private = 0, // like normal cpu or mtl
    Shared = 1, // this is like cpu shared across multiple cpu's, or metal unified memory
    Concatenated = 2 // this is a memory layout that the user can't really used, but is used when multiple tensors are concatenated
};

constexpr DeviceType dCPU = DeviceType::CPU;
constexpr DeviceType dMETA = DeviceType::META;
constexpr DeviceType dMTL = DeviceType::MTL;

inline std::ostream& operator << (std::ostream& os, const DeviceType& dt) noexcept{
	switch(dt){
		case DeviceType::CPU:
			return os << "DeviceType::CPU";
        case DeviceType::MTL:
            return os << "DeviceType::MTL";
		default:
			return os << "UnknownDevice";
	}
}

inline std::ostream& operator << (std::ostream& os, const MemoryLayout& ml) noexcept{
	switch(ml){
		case MemoryLayout::Private:
			return os << "MemoryLayout::Private";
        case MemoryLayout::Shared:
            return os << "MemoryLayout::Shared";
        case MemoryLayout::Concatenated:
            return os << "MemoryLayout::Concatenated";
        case MemoryLayout::None:
            return os << "MemoryLayout::None";
		default:
			return os << "UnknownMemoryLayout";
	}
}


}

#endif
