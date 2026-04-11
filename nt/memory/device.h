#ifndef NT_DEVICE_H__
#define NT_DEVICE_H__
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../dtype/DType_enum.h"
#include "../utils/api_macro.h"
#include "../mtl/abstraction.h"
#include "../utils/span.hpp"
#include <iostream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif
#include "DeviceEnum.h"

namespace nt{

inline uint64_t increment_global_storage_id() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

using DeleterFnPtr = void (*)(void*);
NEUROTENSOR_API void deleteNothing(void*);
template<typename T>
inline void untracked_deleteCPPArray(void* ptr){delete[] static_cast<T*>(ptr);}
NEUROTENSOR_API void deleteAlignedArray(void* ptr);

class NEUROTENSOR_API Device : public intrusive_ptr_target{
	static constexpr DeviceType device_type = dMETA;
    static constexpr MemoryLayout layout_type = MemoryLayout::None;
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
        virtual inline const MemoryLayout& get_memory_layout() const noexcept { return layout_type; }
		virtual inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const {return dev->get_memory() == get_memory();}
		virtual inline const bool in_block(const void*) const {return false;}
        virtual inline const int64_t Size() const { return 0; }
        inline const uint64_t& storage_id() const { return storage_id_; }
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
		inline std::string get_name() const override {return "Device::CPU";}
		inline const DeviceType& get_device_type() const noexcept override {return device_type;}
        inline const MemoryLayout& get_memory_layout() const noexcept override { return layout_type; }
		inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const override {return dev->get_memory() == get_memory();}
		inline const bool in_block(const void* ptr) const override{
			if(end_ == nullptr){return false;}
			return ptr >= memory_ && ptr <= end_;
			/* return reinterpret_cast<const uint8_t*>(ptr) >= reinterpret_cast<const uint8_t*>(memory_) && */
			/*  reinterpret_cast<const uint8_t*>(ptr) <= reinterpret_cast<const uint8_t*>(end_); */
		}
		void capture_memory(void* mem, void* end);
		void capture_deleter(DeleterFnPtr);
        inline const int64_t Size() const override { return end_ - memory_; }
	private:
		static constexpr DeviceType device_type = dCPU;
        static constexpr MemoryLayout layout_type = MemoryLayout::Private;
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
		inline std::string get_name() const override {return "Device::CPU::Shared";}
		inline const DeviceType& get_device_type() const noexcept override {return device_type;}
        inline const MemoryLayout& get_memory_layout() const noexcept override { return layout_type; }
		inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const override {return dev->get_memory() == get_memory();}
		inline const bool in_block(const void* ptr) const override{
			if(end_ == nullptr){return false;}
			return ptr >= memory_ && ptr <= end_;
		}
        inline const int64_t Size() const override { return end_ - memory_; }

	private:

		void* memory_;
		void* end_;
#ifdef _WIN32
		HANDLE hMapFile;
#else
		key_t key;
		int shmid;
#endif
		static constexpr DeviceType device_type = dCPU;
        static constexpr MemoryLayout layout_type = MemoryLayout::Shared;

};

#ifdef NT_MTL_SUPPORTED

class NEUROTENSOR_API DeviceMTLPrivate : public Device{
	public:
		DeviceMTLPrivate();
		~DeviceMTLPrivate() override;

		void allocate_memory(const DType dt, const int64_t size) override;
		void release_memory() override;
        inline intrusive_ptr<mtl::utils::MetalBuffer>& get_buffer() { return this->memory; }
        inline const intrusive_ptr<mtl::utils::MetalBuffer>& get_buffer() const { return this->memory; }
		inline void* get_memory() override {return this->memory->contents();}
		inline void* get_end_memory() override {return reinterpret_cast<char*>(this->memory->contents()) + this->size;}
		inline const void* get_memory() const override {return this->memory->contents();}
		inline const void* get_end_memory() const override {return reinterpret_cast<const char*>(this->memory->contents()) + this->size;}
		inline std::string get_name() const override {return "Device::MTL";}
		inline const DeviceType& get_device_type() const noexcept override {return device_type;}
        inline const MemoryLayout& get_memory_layout() const noexcept override { return layout_type; }
        // just basically if they're holding onto the same memory
		inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const override {return dev->memory == this->memory;}
		inline const bool in_block(const void* ptr) const override{
            return false;
		}
        inline const int64_t& Size() const override {return size;}
        void allocate_memory(const int64_t size);
        inline void adjust_type_bytes(size_t new_bytes) noexcept {
            if(memory != nullptr) memory->typeBytes = new_bytes;
        }
	private:
		static constexpr DeviceType device_type = dMTL;
        static constexpr MemoryLayout layout_type = MemoryLayout::Private;
        intrusive_ptr<mtl::abs::MetalBuffer> memory;
		// MTL::Buffer* memory;
        int64_t size;
};

class NEUROTENSOR_API DeviceMTLShared : public Device{
	public:
		DeviceMTLShared();
		~DeviceMTLShared() override;

		void allocate_memory(const DType dt, const int64_t size) override;
		void release_memory() override;
        inline intrusive_ptr<mtl::utils::MetalBuffer>& get_buffer() { return this->memory; }
        inline const intrusive_ptr<mtl::utils::MetalBuffer>& get_buffer() const { return this->memory; }
		inline void* get_memory() override {return this->memory->contents();}
		inline void* get_end_memory() override {return reinterpret_cast<char*>(this->memory->contents()) + this->size;}
		inline const void* get_memory() const override {return this->memory->contents();}
		inline const void* get_end_memory() const override {return reinterpret_cast<const char*>(this->memory->contents()) + this->size;}
		inline std::string get_name() const override {return "DeviceMTLShared";}
		inline const DeviceType& get_device_type() const noexcept override {return device_type;}
        inline const MemoryLayout& get_memory_layout() const noexcept override { return layout_type; }
        // just basically if they're holding onto the same memory
		inline const bool is_same(const nt::intrusive_ptr<Device>& dev) const override {return dev->memory == this->memory;}
		inline const bool in_block(const void* ptr) const override{
            return false;
		}
        inline const int64_t& Size() const override {return size;}
        void allocate_memory(const int64_t size);
        inline bool is_private() const override { return false; }
        inline void adjust_type_bytes(size_t new_bytes) noexcept {
            if(memory != nullptr) memory->typeBytes = new_bytes;
        }

	private:
		static constexpr DeviceType device_type = dMTL;
        static constexpr MemoryLayout layout_type = MemoryLayout::Shared;
        intrusive_ptr<mtl::abs::MetalBuffer> memory;
		// MTL::Buffer* memory;
        int64_t size;
};

namespace mtl{
DeviceMTLPrivate mtl_shared_to_private(const DeviceMTLShared& device, MTL::CommandBuffer* buffer);
DeviceMTLPrivate& mtl_shared_to_private(const DeviceMTLShared& device, MTL::CommandBuffer* buffer, DeviceMTLPrivate& out);
inline DeviceMTLPrivate mtl_shared_to_private(const DeviceMTLShared& device, intrusive_ptr<abs::MetalCommand> cmd){
    return mtl_shared_to_private(device, cmd->cmd);
}
inline DeviceMTLPrivate& mtl_shared_to_private(const DeviceMTLShared& device, intrusive_ptr<abs::MetalCommand> cmd, DeviceMTLPrivate& out){
    return mtl_shared_to_private(device, cmd->cmd, out);
}
DeviceMTLShared mtl_private_to_shared(const DeviceMTLPrivate& device, MTL::CommandBuffer* buffer);
DeviceMTLShared& mtl_private_to_shared(const DeviceMTLPrivate& device, MTL::CommandBuffer* buffer, DeviceMTLShared& out);
inline DeviceMTLShared mtl_private_to_shared(const DeviceMTLPrivate& device, intrusive_ptr<abs::MetalCommand> cmd){
    return mtl_private_to_shared(device, cmd->cmd);
}
inline DeviceMTLShared& mtl_private_to_shared(const DeviceMTLPrivate& device, intrusive_ptr<abs::MetalCommand> cmd, DeviceMTLShared& out){
    return mtl_private_to_shared(device, cmd->cmd, out);
}
DeviceMTLShared cpu_to_mtl_shared(const DeviceCPU& device);
DeviceMTLPrivate cpu_to_mtl_private(const DeviceCPU& device, MTL::CommandBuffer* buffer);
}

#endif

//this is a class that is used to hold a list of devices
//mainly to support bucket views, and is simple and supports the intrusive_ptr layout
class NEUROTENSOR_API DeviceHolder : public intrusive_ptr_target{
	intrusive_ptr<Device>* devices;
    uint64_t storage_id_;
	public:
		DeviceHolder() = delete;
		explicit DeviceHolder(uint64_t num) 
            : devices(MetaNewArr(intrusive_ptr<Device>,num)),
                storage_id_(increment_global_storage_id())
        {}
		inline ~DeviceHolder() {MetaFreeArr<intrusive_ptr<Device>>(devices);}
		template<typename IntegerType, typename std::enable_if<std::is_integral<IntegerType>::value, int>::type = 0>
        inline intrusive_ptr<Device>& get(IntegerType i) noexcept {return devices[i]};
		template<typename IntegerType, typename std::enable_if<std::is_integral<IntegerType>::value, int>::type = 0>
        inline const intrusive_ptr<Device>& get(IntegerType i) const noexcept {return devices[i]};
		template<typename IntegerType, typename std::enable_if<std::is_integral<IntegerType>::value, int>::type = 0>
		inline intrusive_ptr<Device>& operator[](IntegerType i){return devices[i];} 
		template<typename IntegerType, typename std::enable_if<std::is_integral<IntegerType>::value, int>::type = 0>
		inline const intrusive_ptr<Device>& operator[](IntegerType i) const {return devices[i];} 
        inline intrusive_ptr<DeviceHolder> span(int64_t start, int64_t end) const {
            intrusive_ptr<DeviceHolder> n_devices = make_intrusive<DeviceHolder>(end - start);
            std::copy(devices + start, devices + end, n_devices.devices);
            return std::move(n_devices);
        }
};

NEUROTENSOR_API nt::intrusive_ptr<Device> make_device_(const DeviceType, const MemoryLayout layout = MemoryLayout::Private);
NEUROTENSOR_API nt::intrusive_ptr<DeviceHolder> make_device(const DeviceType, const MemoryLayout layout = MemoryLayout::Private);
NEUROTENSOR_API DeviceType get_device_type(const intrusive_ptr<DeviceHolder>&);
NEUROTENSOR_API MemoryLayout get_memory_layout(const intrusive_ptr<DeviceHolder>&);



}

#endif // NT_DEVICE_H__
