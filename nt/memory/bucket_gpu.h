#ifndef NT_MEMORY_BUCKET_GPU_H__
#define NT_MEMORY_BUCKET_GPU_H__

// this is an abstraction over bucket for explicitly gpu based functions
#include "bucket.h"
#include "../utils/span.hpp"

class BucketGPU : public Bucket {
    protected:
        bool fusing;
    public:
        BucketGPU() = delete;
        BucketGPU(DType dt)
            :Bucket(dt), fusing(false) {}
        BucketGPU(DType dt, bool fusing)
        {}
        inline bool is_gpu() const noexcept override {return true;}
        virtual inline intrusive_ptr<Bucket> new_bounds(int64_t offset, 
                utils::span<int64_t> sizes_, utils::span<int64_t> strides_) const = 0;
        virtual inline intrusive_ptr<Bucket> new_bounds(int64_t* ptr, int64_t size, bool fix = true) const = 0;
        inline intrusive_ptr<Bucket> new_bounds(utils::span<int64_t> span, bool fix = true) const {
            return this->new_bounds(span.data(), span.size(), fix);
        }
        virtual inline bool is_concatenated() const noexcept {return false;}
        inline intrusive_ptr<Bucket> to_mtl(MemoryLayout mem_t = MemoryLayout::Private) const override{
            return this->to_cpu(MemoryLayout::Private)->to_mtl(mem_t);
        }
        inline const bool& is_fusing() const noexcept {return this->fusing;}
        inline void fuse() {
            if(this->fusing){
                // do something to fuse the kernel function's to evaluate this bucket
                this->fusing = false;
            }
        }
        inline bool throw_null(){
            if(this->fusing){
                this->fuse();
                return true;
            }
            if(this->is_null()) return true;
            return false;
        }

};


#endif
