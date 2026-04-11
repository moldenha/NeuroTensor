#include "bucket.h"
#include "bucket_cpu.h"
#include "bucket_mtl.h"
#include "../Tensor.h"
#include "../mtl/utils.h"

namespace nt{

template<>
Tensor Bucket::split<Tensor>(const intrusive_ptr<Bucket>& bkt, uint64_t splitting){
    if(bkt->is_cpu()){
        return intrusive_ptr<BucketCPU>(bkt)->split<Tensor>(splitting);
    }
#ifdef NT_MTL_SUPPORTED
    if(bkt->is_mtl()){
        return intrusive_ptr<BucketMTL>(bkt)->split<Tensor>(splitting);
    }
#endif
    return Tensor::Null();

}


template<>
Tensor Bucket::range<Tensor>(const intrusive_ptr<Bucket>& bkt, std::vector<std::pair<int64_t, int64_t>> ranges){
    if(bkt->is_cpu()){
        return intrusive_ptr<BucketCPU>(bkt)->range<Tensor>(std::move(ranges));
    }
#ifdef NT_MTL_SUPPORTED
    if(bkt->is_mtl()){
        return intrusive_ptr<BucketMTL>(bkt)->range<Tensor>(std::move(ranges));
    }
#endif
    return Tensor::Null();

}

intrusive_ptr<Bucket> Bucket::make_bucket(const int64_t& size, DType dt, DeviceType dev_t, MemoryLayout memory_layout){
    if(dev_t == DeviceType::CPU){
        return make_intrusive<BucketCPU>(size, dt, memory_layout);
    }
    else if(dev_t == DeviceType::MTL){
#ifdef NT_MTL_SUPPORTED
        return make_intrusive<BucketMTL>(size, dt, memory_layout);
#else
        utils::throw_exception(mtl::supported(),
                "Error, Apple's Metal device type $ is not supported on your device",
                dev_t);
#endif
    }
    return make_intrusive<Bucket>();
}

}
