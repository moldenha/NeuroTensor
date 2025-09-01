#ifdef USE_PARALLEL
#include <sys/shm.h>
#include <sys/ipc.h>
#endif

#include "device.h"
#include "../utils/utils.h"
#include "../dtype/DType.h"
#include <iostream>

namespace nt{
DeviceSharedCPU::DeviceSharedCPU()
	:memory_(nullptr), end_(nullptr), key(IPC_PRIVATE), shmid(-1)
{
#ifndef USE_PARALLEL
	utils::THROW_EXCEPTION(false, "Trying to use shared device, but was compiled without parallel capabilities");
#endif
}

DeviceSharedCPU::~DeviceSharedCPU(){
	release_memory();
}


void DeviceSharedCPU::allocate_memory(const DType dt, const int64_t size){
	utils::throw_exception(size >= 0, "Cannot allocate negative bytes of memory, tried to allocate $ bytes", size);
	release_memory();
#ifndef USE_PARALLEL
	utils::THROW_EXCEPTION(false, "Trying to allocate memory on shared device, but was compiled without parallel capabilities");
#else
	key = IPC_PRIVATE;
	const std::size_t byte_size = DTypeFuncs::size_of_dtype(dt);
	const uint64_t n_size = size * byte_size;
	utils::CheckAllocation(DeviceType::CPUShared, n_size);
	this->shmid = shmget(key, n_size, IPC_CREAT | 0666);
	utils::throw_exception(shmid != -1, "Making segment ID failed for shared memory (shmget)");
	memory_ = shmat(shmid, nullptr, 0);
	utils::throw_exception(memory_ != (void*)-1, "Making shared memory failed (shmat)");
	end_ = reinterpret_cast<uint8_t*>(memory_) + n_size;

#endif

}

void DeviceSharedCPU::release_memory(){
#ifdef USE_PARALLEL
	if(memory_){
		utils::DeallocateMemory(DeviceType::CPUShared, static_cast<int64_t>(
				reinterpret_cast<uint8_t*>(end_) -
				reinterpret_cast<uint8_t*>(memory_)));
		shmdt(memory_);
		shmctl(shmid, IPC_RMID, nullptr);
	}
#endif
	memory_ = nullptr;
	end_ = nullptr;
}

}
