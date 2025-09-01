#ifndef NT_DEVICE_H__
#define NT_DEVICE_H__
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../dtype/DType_enum.h"
#include "../utils/api_macro.h"
#include <iostream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif
#include "DeviceEnum.h"

namespace nt{



using DeleterFnPtr = void (*)(void*);
NEUROTENSOR_API void deleteNothing(void*);
template<typename T>
inline void untracked_deleteCPPArray(void* ptr){delete[] static_cast<T*>(ptr);}
NEUROTENSOR_API void deleteAlignedArray(void* ptr);

class NEUROTENSOR_API Device : public intrusive_ptr_target{
	static constexpr DeviceType device_type = dMETA;
	public:
		virtual ~Device() = default;
		virtual void allocate_memory(const DType dt, const int64_t size) = 0;
		virtual void release_memory() = 0;
		virtual void* get_memory() = 0;
		virtual void* get_end_memory() = 0;
		virtual const void* get_memory() const = 0;
		virtual const void* get_end_memory() const = 0;
		virtual inline std::string get_name() const {return "UnkownDevice";}
		virtual inline const DeviceType& get_device_type() const noexcept {return device_type;}
		virtual inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const {return dev->get_memory() == get_memory();}
		virtual inline const bool in_block(const void*) const {return false;}
};


class NEUROTENSOR_API DeviceCPU : public Device{
	public:
		DeviceCPU();
		~DeviceCPU() override;

		void allocate_memory(const DType dt, const int64_t size) override;
		void release_memory() override;
		inline void* get_memory() override {return memory_;}
		inline void* get_end_memory() override {return end_;}
		inline const void* get_memory() const override {return memory_;}
		inline const void* get_end_memory() const override {return end_;}
		inline std::string get_name() const override {return "DeviceCPU";}
		inline const DeviceType& get_device_type() const noexcept override {return device_type;}
		inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const override {return dev->get_memory() == get_memory();}
		inline const bool in_block(const void* ptr) const override{
			if(end_ == nullptr){return false;}
			return ptr >= memory_ && ptr <= end_;
			/* return reinterpret_cast<const uint8_t*>(ptr) >= reinterpret_cast<const uint8_t*>(memory_) && */
			/*  reinterpret_cast<const uint8_t*>(ptr) <= reinterpret_cast<const uint8_t*>(end_); */
		}
		void capture_memory(void* mem, void* end);
		void capture_deleter(DeleterFnPtr);
	private:
		static constexpr DeviceType device_type = dCPU;
		void* memory_;
		void* end_;
		DeleterFnPtr dealc;
};


class NEUROTENSOR_API DeviceSharedCPU : public Device{
	public:
		DeviceSharedCPU();
		~DeviceSharedCPU() override;

		void allocate_memory(const DType dt, const int64_t size) override;
		void release_memory() override;
		inline void* get_memory() override {return memory_;}
		inline void* get_end_memory() override {return end_;}
		inline const void* get_memory() const override {return memory_;}
		inline const void* get_end_memory() const override {return end_;}
		inline std::string get_name() const override {return "DeviceSharedCPU";}
		inline const DeviceType& get_device_type() const noexcept override {return device_type;}
		inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const override {return dev->get_memory() == get_memory();}
		inline const bool in_block(const void* ptr) const override{
			if(end_ == nullptr){return false;}
			return ptr >= memory_ && ptr <= end_;
		}

	private:

		void* memory_;
		void* end_;
#ifdef _WIN32
		HANDLE hMapFile;
#else
		key_t key;
		int shmid;
#endif
		static constexpr DeviceType device_type = dCPUShared;

};

//this is a class that is used to hold a list of devices
//mainly to support bucket views, and is simple and supports the intrusive_ptr layout
class NEUROTENSOR_API DeviceHolder : public intrusive_ptr_target{
	intrusive_ptr<Device>* devices;
	public:
		DeviceHolder() = delete;
		explicit DeviceHolder(uint64_t num) : devices(MetaNewArr(intrusive_ptr<Device>,num)) {}
		inline ~DeviceHolder() {MetaFreeArr<intrusive_ptr<Device>>(devices);}
		template<typename IntegerType, typename std::enable_if<std::is_integral<IntegerType>::value, int>::type = 0>
		inline intrusive_ptr<Device>& operator[](IntegerType i){return devices[i];} 
		template<typename IntegerType, typename std::enable_if<std::is_integral<IntegerType>::value, int>::type = 0>
		inline const intrusive_ptr<Device>& operator[](IntegerType i) const {return devices[i];} 

};

NEUROTENSOR_API nt::intrusive_ptr<Device> make_device(const DeviceType);
NEUROTENSOR_API DeviceType get_device_type(const intrusive_ptr<DeviceHolder>&);



}

#endif // NT_DEVICE_H__
