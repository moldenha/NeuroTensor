#ifndef NT_BUCKET_MTL_H__
#ifdef NT_MTL_SUPPORTED
#define NT_BUCKET_MTL_H__

#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../intrusive_ptr/intrusive_tracked_list.hpp"
#include "device.h"
#include "bucket.h"
#include "bucket_gpu.h"
#include <memory>
/* #include "../dtype/DType.h" */
#include "../utils/utils.h"
#include <vector>
#include "../dtype/DType_enum.h"
#include <functional>
#include "iterator.h"
#include <type_traits>
#include "../utils/api_macro.h"
#include "meta_allocator.h"
#include <variant>
#include "span.hpp"
/* #include "../dtype/ArrayVoid.h" */

namespace nt{

// this is the different buckets available

class BucketMTL;

}

#include "bucket_cpu.h"

namespace nt{

namespace mtl{
void synchronize(const BucketMTL&);
intrusive_ptr<Bucket> synchronize(const intrusive_ptr<BucketMTL>&);
namespace abs::encoder_tensor_details{

void EncodeMTLBucket(intrusive_ptr<MetalArgEncoder> args,
                    intrusive_ptr<BucketMTL> bucket,
                    const int64_t& binding_index,
                    NS::Array* bindings,
                    intrusive_ptr<MetalCommand> cmd,
                    int64_t concat_index);


}
}

class NEUROTENSOR_API BucketMTL : public BucketGPU{
        template <class TTarget,
              class DeleteOp,
              class NullType,
              class... Args>
    friend intrusive_ptr<TTarget, DeleteOp, NullType>
    make_intrusive(Args&&... args);
    friend void mtl::synchronize(const BucketMTL&);
    friend intrusive_ptr<Bucket> mtl::synchronize(const intrusive_ptr<BucketMTL>&);
    friend void mtl::abs::encoder_tensor_details::EncodeMTLBucket(intrusive_ptr<MetalArgEncoder>,
                    intrusive_ptr<BucketMTL>,
                    const int64_t&,
                    NS::Array*,
                    intrusive_ptr<MetalCommand>,
                    int64_t);

    // these are the different bucket kinds
    // this is for a bucket holding contiguous memory
    // this is what is wanted in most cases, (nice and easy)
    struct BucketMTLContiguous {
        intrusive_ptr<DeviceHolder> storage_; // DeviceMTLPrivate || DeviceMTLShared
        inline const intrusive_ptr<Device>& storage() const noexcept { 
            return storage_->get(0);
        }
        inline intrusive_ptr<Device>& storage() noexcept { 
            return storage_->get(0);
        } 
        int64_t offset;
        int64_t numel;
        BucketMTLContiguous()
            :storage_(nullptr), offset(0), nummel(0) 
            {}
    };


    // This affine view is for dense and strided tensors
    // No buckets. No indices. Just math (faster on a gpu).
    // This is designed to support seamlessly:
    /*
     - transpose
     - slice
     - broadcast
     - permute
     - as_strided
    */
    // mtl max dims is 8 -> if more, then switching to dynamic
    struct BucketMTLAffine {
        intrusive_ptr<DeviceHolder> storage_; // DeviceMTLPrivate || DeviceMTLShared
        int64_t offset;
        uint8_t ndim;
        intrusive_ptr<intrusive_tracked_list_sub<int64_t, false>> intrusive_sizes;
        intrusive_ptr<intrusive_tracked_list_sub<int64_t, false>> intrusive_strides;
        BucketMTLAffine()
            :storage_(nullptr),
            offset(0),
            ndim(0),
            intrusive_sizes(nullptr),
            intrusive_strides(nullptr)
            {}

        const int64_t* sizes() const noexcept {
            return intrusive_sizes->get();
        }

        const int64_t* strides() const noexcept {
            return intrusive_strides->get();
        }

        int64_t numel() const noexcept {
            const int64_t* s = this->sizes();
            int64_t o = 1;
            for(uint8_t i = 0; i < ndim; ++i)
                o *= s[i];
            return o;
        }
        inline const intrusive_ptr<Device>& storage() const noexcept { 
            return storage_->get(0);
        }
        inline intrusive_ptr<Device>& storage() noexcept { 
            return storage_->get(0);
        } 


    };

    // This is like the CPU version of strided
    // Where each element has it's own specific index
    //
    struct BucketMTLStrided {
        intrusive_ptr<DeviceHolder> storage_; // DeviceMTLPrivate || DeviceMTLShared
        intrusive_ptr<DeviceMTLPrivate> indexes; // int64_t indexes
        int64_t idx_offset;
        int64_t nnz; // number of non-zero elements
        BucketMTLStrided()
            :storage_(nullptr),
            indexes(nullptr),
            idx_offset(0),
            nnz(0)
        {}
        inline const intrusive_ptr<Device>& storage() const noexcept { 
            return storage_->get(0);
        }
        inline intrusive_ptr<Device>& storage() noexcept { 
            return storage_->get(0);
        } 

    };
    
    struct BucketMTLConcatenated{
        intrusive_ptr<DeviceHolder> devices;
        std::vector<mtl::abs::MetalBufferView> buffers;
        int64_t total_numel;
    };

    std::variant<BucketMTLContiguous,
                 BucketMTLAffine,
                 BucketMTLStrided,
                 BucketMTLConcatenated> storage;

    // storage.index() == 0 : contiguous
    // storage.index() == 1 : affine
    // storage.index() == 2 : strided
    // storage.index() == 3 : concatenated
    
    std::string determine_kernel_name(std::string start, bool vectorized=false, int index = 0) const;
    inline intrusive_ptr<Device>& get_device(){
        if(stoage.index() == 0)
            return std::get<0>(storage).storage();
        if(storage.index() == 1)
            return std::get<1>(storage).storage();
        if(storage.index() == 2)
            return std::get<2>(storage).storage();
        if(storage.index() == 3)
            return std::get<3>(storage).devices->get(0);
    }
    inline const intrusive_ptr<Device>& get_device() const {
        if(stoage.index() == 0)
            return std::get<0>(storage).storage();
        if(storage.index() == 1)
            return std::get<1>(storage).storage();
        if(storage.index() == 2)
            return std::get<2>(storage).storage();
        if(storage.index() == 3)
            return std::get<3>(storage).devices->get(0);
    }


    std::vector<mtl::abs::MetalBufferView>& emplace_metal_buffer_view(std::vector<mtl::abs::MetalBufferView>&) const;
    mtl::abs::MetalBufferView get_metal_buffer_view() const;

    inline intrusive_ptr<mtl::abs::MetalBuffer> get_buffer() noexcept {
        intrusive_ptr<Device> dev = this->get_device();
        if(dev->is_private()){
            return intrusive_ptr<DeviceMTLPrivate>(dev)->get_buffer();
        }
        return intrusive_ptr<DeviceMTLShared>(dev)->get_buffer();
    }
    int64_t byte_size() const noexcept;
    BucketMTL(BucketMTLContiguous, DType);
    BucketMTL(BucketMTLAffine, DType);
    BucketMTL(BucketMTLStrided, DType);
    BucketMTL(BucketMTLConcatenated, DType);
    BucketMTL(const mtl::abs::MetalBufferView&, const intrusive_ptr<DeviceHolder>&, DType);
    int64_t offset(bool strided = true) const;
    BucketMTL& clone(BucketMTL&) const;
    template<typename T>
	T split_strided_(uint64_t splitting) const;
	template<typename T>
	T split_contiguous_(uint64_t splitting) const;
	template<typename T>
	T split_bucketed_(uint64_t splitting) const;
	template<typename T>
	T split_concatenated_(uint64_t splitting) const;
	template<typename T>
	T range_concatenated_(std::vector<std::pair<int64_t, int64_t>> ranges) const;
    BucketMTL new_bounds_concatenated_(int64_t* ptr, int64_t size) const;
    BucketMTL bucket_all_indices_concatenated() const;

    public:
        BucketMTL();
        BucketMTL(const int64_t size, DType dt, MemoryLayout memory_layout = MemoryLayout::Private);
        BucketMTL(const BucketMTL& b);
        BucketMTL(BucketMTL&& b);
        BucketMTL(std::nullptr_t);
        BucketMTL(std::nullptr_t, bool);
        BucketMTL& operator=(const BucketMTL& b);
        BucketMTL& operator=(BucketMTL&& b);
        inline bool is_contiguous() const noexcept override {
            return storage.index() == 0;
        }
        inline bool is_affine() const noexcept override {
            return storage.index() == 1;
        }
        inline bool is_strided() const noexcept override {
            return storage.index() == 2;
        }
        inline bool is_concatenated() const noexcept override {
            return storage.index() == 3;
        }
        inline void nullify() override{
            storage = 
                BucketMTLContiguous();
        }
        inline int64_t storage_size() const override {
            if(this->is_concatenated()){
                return this->numel() * this->byte_size();
            }
            return this->get_device()->Size();
        }

        inline int64_t numel() const noexcept override {
            if(this->is_null()) return 0;
            if(storage.index() == 0)
                return std::get<0>(storage).numel;
            if(storage.index() == 1)
                return std::get<1>(storage).numel();
            if(storage.index() == 2)
                return std::get<2>(storage).nnz;
            if(storage.index() == 3)
                return std::get<3>(storage).total_numel;
        }
        inline std::vector<int64_t> get_numels(bool add_prev = false) const noexcept {
            if(storage.index() != 3) return std::vector<int64_t>{this->numel()};
            // for concatenation it returns all the different numel's
            const auto& s = std::get<3>(storage);
            std::vector<int64_t> numels(s.buffers.size());
            int64_t b_size = this->byte_size();
            numels[0] = s.buffers[0].numelBytes / b_size;
            if(add_prev){
                for(size_t i = 1; i < numels.size(); ++i){
                    numels[i] = (s.buffers[i].numelBytes / b_size) + numels[i-1];
                }
                
            }
            else{
                for(size_t i = 1; i < numels.size(); ++i){
                    numels[i] = s.buffers[i].numelBytes / b_size;
                }
            }
            return std::move(numels);
        }

        inline DeviceType device_type() const noexcept override { return DeviceType::MTL; }
        inline MemoryLayout memory_layout() const noexcept override {
            return this->is_concatenated() ? MemoryLayout::Concatenated : this->get_device()->get_memory_layout();
        }

        BucketMTL contiguous_mtl() const;
        inline intrusive_ptr<Bucket> contiguous() const override {
            return make_intrusive<BucketMTL>(this->contiguous_mtl());
        }
        BucketMTL clone_mtl() const;
        inline intrusive_ptr<Bucket> clone() const override {
            return make_intrusive<BucketMTL>(this->clone_mtl());
        }
        inline bool is_shared() const noexcept override {
            return this->memory_layout() == MemoryLayout::Shared;
        }
        inline bool is_private() const noexcept override {
            return this->memory_layout() == MemoryLayout::Private;
        }
        inline bool is_gpu() const noexcept override {return true;}
        inline bool is_cpu() const noexcept override {return false;}
        inline bool is_mtl() const noexcept override {return true;}
        intrusive_ptr<Bucket> to_shared() const override;
        intrusive_ptr<Bucket> to_private() const override;
        intrusive_ptr<Bucket> to_cpu(MemoryLayout mem_t = MemoryLayout::Private) const override;
        inline intrusive_ptr<Bucket> to_mtl(MemoryLayout mem_t = MemoryLayout::Private) const override {return this->to_memory_layout(mem_t);}
        inline int64_t use_count() const override { return this->get_device().use_count(); }
        BucketMTL new_bounds_mtl(int64_t start, int64_t end) const;
        BucketMTL new_bounds_mtl(int64_t offset, utils::span<int64_t> sizes_, utils::span<int64_t> strides_) const;
        // fix can be false
        // however, it would be hard for me to think of a scenario in which fix should ever be false
        BucketMTL new_bounds_mtl(int64_t* ptr, int64_t size, bool fix=true) const; // will have to make one that takes a bucket/private device
        
        inline intrusive_ptr<Bucket> new_bounds(int64_t start, int64_t end) const override {
            return make_intrusive<BucketMTL>(this->new_bounds_mtl(start, end));
        }
        inline intrusive_ptr<Bucket> new_bounds(int64_t offset, utils::span<int64_t> sizes_, utils::span<int64_t> strides_) const override {
            return make_intrusive<BucketMTL>(this->new_bounds_mtl(offset, sizes_, strides_));
        }
        inline intrusive_ptr<Bucket> new_bounds(int64_t offset, utils::span<int64_t> sizes_, utils::span<int64_t> strides_) const override {
            return make_intrusive<BucketMTL>(this->new_bounds_mtl(offset, sizes_, strides_));
        }
        inline const bool is_null() const noexcept override {
            return this->get_device().is_null();
        }

        inline void* data_ptr() noexcept override {
            if(this->is_null() || this->is_private() || this->is_concatenated())
                return nullptr;
            return this->get_device()->get_memory();
        }

        inline void* data_ptr_end() noexcept override {
            if(this->is_null() || this->is_private() || this->is_concatenated())
                return nullptr;
            return this->get_device()->get_end_memory();
        }
        inline const void* data_ptr() const noexcept override {
            if(this->is_null() || this->is_private() || this->is_concatenated())
                return nullptr;
            return this->get_device()->get_memory();
        }

        inline const void* data_ptr_end() const noexcept override {
            if(this->is_null() || this->is_private() || this->is_concatenated())
                return nullptr;
            return this->get_device()->get_end_memory();
        }
        
        // this function basically goes from
        // affine -> strided
        // or:
        // contiguous -> strided
        BucketMTL bucket_all_indices_mtl() const;
        inline intrusive_ptr<Bucket> bucket_all_indices() const override{
            return intrusive_ptr<BucketMT>(this->bucket_all_indices_mtl());
        }

        template<typename T>
        T split(uint64_t splitting) const;
        template<typename T>
        T range(std::vector<std::pair<int64_t, int64_t>> ranges) const;
        void swap(BucketMTL&);
        inline bool can_force_contiguity() const noexcept override {return true;}
        inline bool can_force_contiguity(const int64_t& bytes) const noexcept override {return true;}
        inline int64_t force_contig_size() const noexcept override { return this->get_device()->Size(); }
        intrusive_ptr<Bucket> bound_force_contiguity_bucket() const override;
        intrusive_ptr<Bucket> force_contiguity_and_bucket() const override;
        intrusive_ptr<Bucket> force_contiguity(int64_t) const override;
        intrusive_ptr<Bucket> copy_strides(bool copy_vals = true) const override;
        intrusive_ptr<Bucket> new_stride_size(int64_t size, bool is_blocked = false) const override;
        BucketMTL operator+(int64_t) const;
        BucketMTL operator-(int64_t) const;
        static BucketMTL catV(std::vector<BucketMTL>);
        static BucketMTL from_cpu(const BucketCPU&);
        inline std::vector<uint64_t> storage_id() const noexcept {
            if(stoage.index() == 0)
                return std::get<0>(storage).storage_->storage_id();
            if(storage.index() == 1)
                return std::get<1>(storage).storage_->storage_id();
            if(storage.index() == 2)
                return std::get<2>(storage).storage->storage_id();
            if(storage.index() == 3)
                return std::get<3>(storage).devices->storage_id();

        }
        
};

}


namespace std{
    inline void swap(::nt::BucketMTL& lhs, ::nt::BucketMTL& rhs){
        lhs.swap(rhs);
    }
}

#endif // ifdef NT_MTL_SUPPORTED
#endif // ifndef NT_BUCKET_MTL_H__
