#ifndef NT_MEMORY_BUCKET_H__
#define NT_MEMORY_BUCKET_H__
// the way that this works is through inheritance with the different buckets

// Inheritance Justification:
/*

   Each bucket of memory is used to basically handle memory in it's own specific ways according to the device
   For example, CUDA, Apple MTL, and a standard CPU all have very different ways of handling their memory
        and different ways that it should be optimized
   However, there needs to be some semblence of running functions amongst all of them such that they can run certain
        of the same functions
   Inheritance offers a way to outline what functions need to be synonomous and available for all routes, and
        being able to call them easily

 */

#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "DeviceEnum.h"
#include "../dtype/DTypeEnum.h"
#include "../utils/api_macro.h"

namespace nt{

class NEUROTENSOR_API Bucket : intrusive_ptr_target{
    protected:
        DType dtype_;
    public:
        Bucket() = delete;
        Bucket(DType dt) : dtype_(dt) {}
        Bucket(const Bucket&) = default;
        Bucket(Bucket&&) = default;
        Bucket& operator=(const Bucket&) = default;
        Bucket& operator=(Bucket&&) = default;
        static intrusive_ptr<Bucket> make_bucket(const int64_t& size, DType dt, 
                                                    DeviceType dev_t, 
                                                    MemoryLayout memory_layout = MemoryLayout::Private);
        inline const DType& dtype() const noexcept {return dtype_;}
        virtual inline bool is_contiguous() const noexcept {return false;}
        virtual inline bool is_affine() const noexcept {return false;}
        virtual inline bool is_strided() const noexcept {return false;}
        virtual void nullify() = 0;
        virtual inline int64_t stoage_size() const noexcept {return 0;}
        virtual inline int64_t numel() const noexcept {return 0:}
        virtual inline DeviceType device_type() const noexcept {return DeviceType::META;}
        virtual inline MemoryLayout memory_layout() const noexcept {return MemoryLayout::None;}
        virtual inline intrusive_ptr<Bucket> contiguous() const = 0;
        virtual inline intrusive_ptr<Bucekt> clone() const = 0;
        virtual inline bool is_private() const noexcept {return false;}
        virtual inline bool is_shared() const noexcept {return false;}
        virtual inline intrusive_ptr<Bucket> to_private() const {return make_intrusive<Bucket>(nullptr);}
        virtual inline intrusive_ptr<Bucket> to_shared() const {return make_intrusive<Bucket>(nullptr);}
        inline intrusive_ptr<Bucket> to_memory_layout(const MemoryLayout& mem_t) const {
            switch(mem_t){
                case MemoryLayout::Shared:
                    return this->to_shared();
                case MemoryLayout::Private:
                    return this->to_private();
                default:
                    utils::throw_exception(false, "Cannot explicitly convert memory layout to $", mem_t);
                    return intrusive_ptr<Bucket>(nullptr);
            }
        }
        virtual inline intrusive_ptr<Bucket> to_cpu(MemoryLayout mem_t = MemoryLayout::Private) {return make_intrusive<Bucket>(nullptr);}
        virtual inline intrusive_ptr<Bucket> to_mtl(MemoryLayout mem_t = MemoryLayout::Private) {return make_intrusive<Bucket>(nullptr);}

        virtual inline int64_t use_count() {return 0;}
        virtual inline bool is_gpu() const noexcept {return false;}
        virtual inline bool is_cpu() const noexcept {return false;}
        virtual inline bool is_mtl() const noexcept {return false;}
        inline intrusive_ptr<Bucket> to_device(const DeviceType& dev_t, MemoryLayout mem_t = MemoryLayout::Private){
            switch(dev_t){
                case DeviceType::CPU:
                    return this->to_cpu(mem_t);
                case DeviceType::MTL:
                    return this->to_mtl(mem_t);
                default:
                    utils::throw_exception(false, "Cannot explicitly convert memory layout to $",  dev_t);
                    return intrusive_ptr<Bucket>(nullptr);
            }
        }
        virtual inline intrusive_ptr<Bucket> new_bounds(int64_t start, int64_t end) const = 0;
        virtual inline bool is_null() const noexcept {return true;}
        virtual inline void* data_ptr() noexcept {return nullptr;}
        virtual inline void* data_ptr_end() noexcept {return nullptr;}
        virtual inline const void* data_ptr() const noexcept {return nullptr;}
        virtual inline const void* data_ptr_end() const noexcept {return nullptr;}
        virtual inline intrusive_ptr<Bucket> bucket_all_indices() const = 0;
        inline intrusive_ptr<Bucket> add(int64_t i) const { return this->new_bounds(i, this->numel()); }
        /*
        template<typename T>
        T split(uint64_t splitting) const;
        template<typename T>
        T range(std::vector<std::pair<int64_t, int64_t>> ranges) const;
         */

        template<typename T>
        static T split(const intrusive_ptr<Bucket>&, uint64_t);
        template<typename T>
        static T range(const intrusive_ptr<Bucket>&, std::vector<std::pair<int64_t, int64_t>> ranges);
        
        virtual inline bool can_force_contiguity() const noexcept {return false;}
        virtual inline bool can_force_contiguity(const int64_t& bytes) const noexcept {return false;}
        virtual inline int64_t force_contig_size() const noexcept { return 0; }
        virtual inline intrusive_ptr<Bucket> bound_force_contiguity_bucket() const = 0;
        virtual inline intrusive_ptr<Bucket> force_contiguity_and_bucket() const = 0;
        virtual inline intrusive_ptr<Bucket> force_contiguity(int64_t) const = 0;
        virtual inline intrusive_ptr<Bucket> copy_strides() const = 0;
        virtual inline intrusive_ptr<Bucket> new_stride_size(int64_t, bool is_blocked = false) const = 0;
        virtual inline std::vector<uint64_t> storage_id() const noexcept = 0;
        
};



}



#endif
