#include <windows.h>
#include "device.h"
#include "../utils/utils.h"
#include "../dtype/DType.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{

DeviceSharedCPU::DeviceSharedCPU()
    : memory_(nullptr), end_(nullptr), hMapFile(nullptr)
{
#ifndef USE_PARALLEL
	utils::THROW_EXCEPTION(false, "Trying to use shared device, but was compiled without parallel capabilities");
#endif
}

DeviceSharedCPU::~DeviceSharedCPU() {
	release_memory();
}

void DeviceSharedCPU::allocate_memory(const DType dt, const int64_t size) {
	utils::throw_exception(size >= 0, "Cannot allocate negative bytes of memory, tried to allocate $ bytes", size);
	release_memory();
#ifndef USE_PARALLEL
	utils::THROW_EXCEPTION(false, "Trying to allocate memory on shared device, but was compiled without parallel capabilities");
#else
	const std::size_t byte_size = DTypeFuncs::size_of_dtype(dt);
	const uint64_t n_size = size * byte_size;
	utils::CheckAllocation(DeviceType::CPUShared, n_size);
	hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)(n_size >> 32), (DWORD)(n_size & 0xFFFFFFFF), nullptr);
	utils::throw_exception(hMapFile != nullptr, "Failed to create file mapping object");

	memory_ = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, n_size);
	utils::throw_exception(memory_ != nullptr, "Failed to map view of file");

	end_ = reinterpret_cast<uint8_t*>(memory_) + n_size;
#endif
}

void DeviceSharedCPU::release_memory() {
#ifdef USE_PARALLEL
	if (memory_) {
		utils::DeallocateMemory(DeviceType::CPUShared, static_cast<int64_t>(
					reinterpret_cast<uint8_t*>(end_) -
					reinterpret_cast<uint8_t*>(memory_)));
		UnmapViewOfFile(memory_);
		CloseHandle(hMapFile);
	}
#endif
	memory_ = nullptr;
	end_ = nullptr;
	hMapFile = nullptr;
}

}
