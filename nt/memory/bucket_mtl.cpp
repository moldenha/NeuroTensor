#include "../mtl/abstraction.h"
#include "bucket_mtl.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp> // Only needed if rendering; safe to ignore here

#include "devices.h"
#include "meta_allocator.h"
#include "../utils/vla_macro.h"
#include "../mtl/functional/iota.h"
#include "../dtype/dtype.h"
#include "../Tensor.h"
#include "bucket_cpu.h"
#include "../utils/span.hpp"

namespace nt{


BucketMTL::BucketMTL()
    :BucketGPU(DType::Float32),
    storage(BucketMTLContiguous({make_device(dMTL), 0, 1}))
    {
        auto& contig = std::get<0>(this->storage);
        contig->allocate_memory(DType::Float32, 1);
    }

BucketMTL::BucketMTL(BucketMTLContiguous stroage_, DType dtype_)
    :BucketGPU(dtype_),
    storage(storage_)
{}

BucketMTL::BucketMTL(BucketMTLAffine stroage_, DType dtype_)
    :BucketGPU(dtype_),
    storage(storage_)
{}

BucketMTL::BucketMTL(BucketMTLStrided stroage_, DType dtype_)
    :BucketGPU(dtype_),
    storage(storage_)
{}

BucketMTL::BucketMTL(BucketMTLConcatenated stroage_, DType dtype_)
    :BucketGPU(dtype_),
    storage(storage_)
{}

BucketMTL::BucketMTL(const mtl::abs::MetalBufferView& view, const intrusive_ptr<DeviceHolder>& device, DType dtype_)
    :BucketGPU(dtype_)
{
    if(view.indexes != nullptr){
        storage = BucketMTLStrided{
            .storage_ = device,
            .indexes = view.indexes,
            .idx_offset = view.idxOffset,
            .nnz = view.numelBytes / view.buffer->typeBytes
        };
    }else if(view.sizes != nullptr && view.strides != nullptr){
        storage = BucketMTLAffine{
            .storage_ = device,
            .offset = view.offsetBytes / view.buffer->typeBytes,
            .ndim = view.ndim,
            .intrusive_sizes = view.sizes,
            .inrusive_strides = view.strides
        };
    }else if(view.indexes == nullptr && view.sizes == nullptr && view.strides == nullptr){
        storage = BucketMTLContiguous{
            .storage_ = device,
            .offset = view.offsetBytes / view.buffer->typeBytes,
            .numel = view.numelBytes / view.buffer->typeBytes,
        };
    }else{
        utils::THROW_EXCEPTION(false,
                "Error: Constructor got invalid buffer view");
    }
}

BucketMTL::BucketMTL(const int64_t size, DType dt, MemoryLayout layout)
    :BucketGPU(dt),
    storage(BucketMTLContiguous({make_device(dMTL, layout), 0, size}))
{
    auto& contig = std::get<0>(this->storage);
    contig->storage()->allocate_memory(dt, size);
}

BucketMTL::BucketMTL(const BucketMTL& b)
    :BucketGPU(b.dtype_),
    storage(b.storage)
{}

BucketMTL::BucketMTL(BucketMTL&& b)
    :BucketGPU(b.dtype_),
    storage(std::move(b.storage))
{}

BucketMTL::BucketMTL(std::nullptr_t)
    :BucketGPU(DType::Float32),
    storage(BucketMTLContiguous({nullptr, 0, 0})),
{}

BucketMTL::BucketMTL(std::nullptr_t, bool fusing)
    :BucketGPU(DType::Float32, fusing),
    storage(BucketMTLContiguous({nullptr, 0, 0})),
{}


BucketMTL& BucketMTL::operator=(const BucketMTL& b){
    this->storage = b.storage;
    this->dtype_ = b.dtype_;
    return *this;
}

BucketMTL& BucketMTL::operator=(BucketMTL&& b){
    this->storage = std::move(b.storage);
    this->dtype_ = b.dtype_;
    return *this;
}

mtl::abs::MetalBufferView BucketMTL::get_metal_buffer_view() const {
    utils::throw_exception(!this->is_null(), "Cannot get metal buffer view of null memory");
    using namespace mtl::abs;
    std::size_t dtype_size = DTypeFuncs::size_of_dtype(this->dtype_); 
    if(this->is_contiguous()){
        auto& s = std::get<0>(this->storage);
        return MetalBufferView{
                .buffer = this->get_buffer(),
                .offsetBytes = s.offset * dtype_size,
                .numelBytes = s.numel * dtype_size,
                .idxOffset = 0,
                .ndim = 0,
                .sizes = nullptr,
                .strides = nullptr,
                .indexes = nullptr
        };
    }
    else if(this->is_affine()){
        return MetalBufferView{
                .buffer = this->get_buffer(),
                .offsetBytes = s.offset * dtype_size,
                .numelBytes = s.numel() * dtype_size,
                .idxOffset = 0,
                .typeBytes = dtype_size,
                .ndim = s.ndim,
                .sizes = s.intrusive_sizes,
                .strides = s.intrusive_strides,
                .indexes = nullptr
        }; 
    }
    else if(this->is_strided()){
        return MetalBufferView{
                .buffer = this->get_buffer(),
                .offsetBytes = 0,
                .numelBytes = s.nnz * dtype_size,
                .idxOffset = s.idx_offset,
                .typeBytes = dtype_size,
                .ndim = 0,
                .sizes = 0,
                .strides = 0,
                .indexes = s.indexes
        }; 
    }
    else if(this->is_concatenated()){
        return std::get<3>(this->storage).buffers[0];
    }
}

std::vector<mtl::abs::MetalBufferView>& BucketMTL::emplace_metal_buffer_view(std::vector<mtl::abs::MetalBufferView>& vec) const{
    utils::throw_exception(!this->is_null(), "Cannot get metal buffer view of null memory");
    using namespace mtl::abs;
    std::size_t dtype_size = DTypeFuncs::size_of_dtype(this->dtype_); 
    if(this->is_contiguous()){
        auto& s = std::get<0>(this->storage);
        vec.emplace_back(MetalBufferView{
                .buffer = this->get_buffer(),
                .offsetBytes = s.offset * dtype_size,
                .numelBytes = s.numel * dtype_size,
                .idxOffset = 0,
                .ndim = 0,
                .sizes = nullptr,
                .strides = nullptr,
                .indexes = nullptr
        });
        return vec;
    }else if(this->is_affine()){
        auto& s = std::get<1>(this->storage);
        vec.emplace_back(MetalBufferView{
                .buffer = this->get_buffer(),
                .offsetBytes = s.offset * dtype_size,
                .numelBytes = s.numel() * dtype_size,
                .idxOffset = 0,
                .typeBytes = dtype_size,
                .ndim = s.ndim,
                .sizes = s.intrusive_sizes,
                .strides = s.intrusive_strides,
                .indexes = nullptr
        });
        return vec;
    }else if(this->is_strided()){
        auto& s = std::get<2>(this->storage);
        vec.emplace_back(MetalBufferView{
                .buffer = this->get_buffer(),
                .offsetBytes = 0,
                .numelBytes = s.nnz * dtype_size,
                .idxOffset = s.idx_offset,
                .typeBytes = dtype_size,
                .ndim = 0,
                .sizes = 0,
                .strides = 0,
                .indexes = s.indexes
        });
        return vec;
    }else if(this->is_concatenated()){
        auto& s = std::get<3>(this->storage);
        for(const auto& view : s.buffers){
            vec.emplace_back(view);
        }
        return vec;
    }
    return vec;

}

int64_t BucketMTL::byte_size() const noexcept {return DTypeFuncs::size_of_dtype(this->dtype_);}

// example input: "clone_"
std::string BucketMTL::determine_kernel_name(std::string start, bool vectorized, int index) const{
    if(this->is_contiguous())
        start += "contiguous_kernel_";
    else if(this->is_affine()){
        start += "affine_kernel_";
        vectorized = false;
    }
    else if(this->is_strided()){
        start += "strided_kernel_";
        vectorized = false;
    }else if(this->is_concatenated()){
        if(index == 0)
            start += "contiguous_kernel_";
        else if(index == 1)
            start += "affine_kernel_";
        else if(index == 2)
            start += "strided_kernel_";
        vectorized = false;

    }
    switch(this->dtype_){
        case DType::Float32:
            if(vectorized)
                return start + "float4";
            return start + "float";
        case DType::Float16:
            if(vectorized)
                return start + "half4";
            return start + "half";
        case DType::int8:
            return start + "char";
        case DType::uint8:
            return start + "uchar";
        case DType::int16:
            return start + "short";
        case DType::uint16:
            return start + "ushort";
        case DType::int32:
            return start + "int";
        case DType::uint32:
            return start + "uint";
        case DType::int64:
            return start + "long";
        case DType::Complex64:
            return start + "float2";
        case DType::Complex32:
            return start + "half2";
        case DType::Bool:
            return start + "bool";
        default:
            utils::throw_exception(false,
                    "Error unknown mtl dtype support for $",
                    this->dtype_);
            return std::move(start);
    }
}

inline int64_t ceil_div(int64_t numerator, int64_t denominator) {
    if (denominator == 0) {
        // Handle division by zero error as appropriate for your application
        // This example returns a specific error code or can throw an exception
        return 0; 
    }

    if (denominator < 0) {
        // Ensure denominator is positive for standard formula
        numerator = -numerator;
        denominator = -denominator;
    }

    if (numerator >= 0) {
        return (numerator + denominator - 1) / denominator;
    } else {
        return numerator / denominator;
    }
}

// if vectorized -> ceil_div(numel, 4) 
// otherwise nothing to account for
// this is used when calculating threads, 
//  otherwise, the number of elements will go over
inline int64_t vectorized_div(int64_t numel, bool vectorized){
    return (vectorized) ? (numel + 3) / 4 : numel;
}

BucketMTL& BucketMTL::clone(BucketMTL& out_bucket) const {
    utils::THROW_EXCEPTION(!this->is_null() && !out_bucket.is_null(), "Cannot clone null memory into null memory");
    utils::THROW_EXCEPTION(
            out_bucket.is_contiguous(),
            "INTERNAL ERROR: Expected out bucket from internal clone to be contiguous");
    utils::THROW_EXCEPTION(
            out_bucket.dtype_ == this->dtype_,
            "INTERNAL ERROR: Expected out bucket from internal clone to have same dtype ($ != !)",
            out_bucket.dtype_, this->dtype_);
    utils::THROW_EXCEPTION(
            out_bucket.numel() == this->numel(),
            "INTERNAL ERROR: Expected  out bucket from internal clone to have same numel ($ != !)",
            out_bucket.numel(), this->numel());
    //loading the shader library
    mtl::abs::MetalContext& ctx = mtl::abs::MetalContext::instance();
    



    // MTL::CommandQueue* queue = ctx.queue();
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = ctx.makeCommandBuffer();
    int64_t dtype_size = static_cast<int64_t>(DTypeFuncs::size_of_dtype(this->dtype_));
    int64_t numel = this->numel();
    if(this->is_contiguous()){
        std::string kernelName = this->determine_kernel_name("clone_", /* vectorize = */ true);
        bool vectorized = (kernelName.back() == '4'); // determines that the function is going to be vectorized
        intrusive_ptr<mtl::abs::Pipeline> pipeline_ = ctx.get_pipeline(kernelName);
        const int64_t nums = vectorized_div(numel, vectorized);
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = nums,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{this->get_buffer(), std::get<0>(storage).offset * dtype_size},
                mtl::abs::utils::EncodeBuffer{out_bucket.get_buffer(), std::get<0>(out_bucket.storage).offset * dtype_size},
                nums
        );
    }else if (this->is_affine()){
        std::string kernelName = this->determine_kernel_name("clone_", /* vectorize = */ false);
        intrusive_ptr<mtl::abs::Pipeline> pipeline_ = ctx.get_pipeline(kernelName);
        const auto& a = std::get<1>(this->storage);
        uint32_t ndim = a.ndim;
        // make sure ndim * sizeof(int64_t) < 2000
        // therefore, ndim is less than 250
        //  (which it should never excede, that would be rediculous)

        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = numel,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{this->get_buffer(), a.offset * dtype_size},
                mtl::abs::utils::EncodeBuffer{out_bucket.get_buffer(), std::get<0>(out_bucket.storage).offset * dtype_size},
                static_cast<uint32_t>(ndim), 
                a.intrusive_sizes, 
                a.intrusive_strides, 
                numel
        ); 
    }else if (this->is_strided()){
        std::string kernelName = this->determine_kernel_name("clone_", /* vectorize = */ false);
        intrusive_ptr<mtl::abs::Pipeline> pipeline_ = ctx.get_pipeline(kernelName);
        const auto& s = std::get<2>(this->storage);
        int64_t nnz = s.nnz;
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = nnz,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{this->get_buffer(), a.offset * dtype_size},
                mtl::abs::utils::EncodeBuffer{out_bucket.get_buffer(), std::get<0>(out_bucket.storage).offset * dtype_size},
                mtl::abs::utils::EncodeBuffer{s.indexes->get_buffer(), s.idx_offset * sizeof(int64_t)},
                nnz
        );

    }else if(this->is_concatenated()){
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer,
                    .pipeline = nullptr,
                    .size = numel,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
            makeEncoderArguments( // contiguous arguments
                ctx.get_pipeline(
                        this->determine_kernel_name("clone_", /* vectorize = */ false, /*index = */ 0)
                ),
                mtl::abs::utils::ViewBufferArg::Buffer,
                mtl::abs::utils::EncodeBuffer{out_buffer.get_buffer(), std::get<0>(out_bucket.storage).offset},
                mtl::abs::utils::ViewBufferArg::Numel
            ),
            makeEncoderArguments( // affine arguments
                ctx.get_pipeline(
                        this->determine_kernel_name("clone_", /* vectorize = */ false, /*index = */ 1)
                ),
                mtl::abs::utils::ViewBufferArg::Buffer,
                mtl::abs::utils::EncodeBuffer{out_buffer.get_buffer(), std::get<0>(out_bucket.storage).offset},
                mtl::abs::utils::ViewBufferArg::Ndim,
                mtl::abs::utils::ViewBufferArg::Sizes,
                mtl::abs::utils::ViewBufferArg::Strides,
                mtl::abs::utils::ViewBufferArg::Numel
            ),
            makeEncoderArguments( // strided arguments
                ctx.get_pipeline(
                        this->determine_kernel_name("clone_", /* vectorize = */ false, /*index = */ 2)
                ),
                mtl::abs::utils::ViewBufferArg::Buffer,
                mtl::abs::utils::EncodeBuffer{out_buffer.get_buffer(), std::get<0>(out_bucket.storage).offset},
                mtl::abs::utils::ViewBufferArg::Indexes,
                mtl::abs::utils::ViewBufferArg::Numel
            ), std::get<3>(this->storage).buffers
        );
    }
    else{
        utils::THROW_EXCEPTION(false,
                "INTERNAL ERROR WITH CLONE: Stride was not detectable");
    }
    ctx.run_command(commandBuffer_);
    return out_bucket;
}

BucketMTL BucketMTL::clone_mtl() const {
    BucketMTL out_bucket(this->numel(), this->dtype_, 
            this->is_concatenated() ? MemoryLayout::Private : this->memory_layout());
    this->clone(out_bucket);
    return std::move(out_bucket);
}

BucketMTL BucketMTL::contiguous_mtl(){
    if(this->is_contiguous())
        return *this;
    return this->clone_mtl();
}

// this is almost verbatim clone
intrusive_ptr<Bucket> BucketMTL::to_shared() const {
    utils::throw_exception(!this->is_null(), "Cannot make null memory shared");
    if(this->is_shared())
        return make_intrusive<BucketMTL>(*this);
    // otherwise it is just a clone, but with a different memory layout
    BucketMTL out_bucket(this->numel(), this->dtype_, MemoryLayout::Shared);
    this->clone(out_bucket);
    return make_intrusive<BucketMTL>(std::move(out_bucket));
}

intrusive_ptr<Bucket> BucketMTL::to_private() const {
    utils::throw_exception(!this->is_null(), "Cannot make null memory private");
    if(this->is_private())
        return make_intrusive<BucketMTL>(*this);
    // otherwise it is just a clone, but with a different memory layout
    BucketMTL out_bucket(this->numel(), this->dtype_, MemoryLayout::Private);
    this->clone(out_bucket);
    return make_intrusive<BucketMTL>(std::move(out_bucket));
}

intrusive_ptr<Bucket> BucketMTL::to_cpu(MemoryLayout mem_t) const {
    utils::throw_exception(!this->is_null(), "Cannot put null memory onto the cpu");
    if(this->is_shared){
        BucketMTL contig = this->contiguous_mtl();
        mtl::synchronize(contig);
        intrusive_ptr<Bucket> out = make_intrusive<BucketCPU>(this->numel(), this->dtype_, mem_t);
        std::memcpy(contig.data_ptr(), out->data_ptr(), this->numel() * this->byte_size());
        return out;
    }
    intrusive_ptr<BucketMTL> contig = this->to_shared();
    mtl::synchronize(*contig);
    intrusive_ptr<Bucket> out = make_intrusive<BucketCPU>(this->numel(), this->dtype_, mem_t);
    std::memcpy(contig->data_ptr(), out->data_ptr(), this->numel() * this->byte_size());
    return out;


}

int64_t BucketMTL::offset(bool strided) const {
    if(this->is_contiguous()){
        return std::get<0>(storage).offset;
    }
    if(this->is_affine()){
        return std::get<1>(storage).offset;
    }
    if(this->is_concatenated())
        return 0;
    if(!strided) return 0;
    // strided
    DeviceMTLShared outBuffer;
    outBuffer.allocate_memory(sizeof(int64_t));
    outBuffer.adjust_type_bytes(sizeof(int64_t));
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = mtl::abs::MetalContext::instance().makeCommandBuffer();
    MTL::CommandBuffer* buffer = commandBuffer_->cmd;
    MTL::BlitCommandEncoder* blit = buffer->blitCommandEncoder();
    blit->copyFromBuffer(std::get<2>(storage).indexes->get_buffer(), 0, outBuffer.get_buffer(), 0, sizeof(int64_t));
    blit->endEncoding();
    mtl::abs::MetalContext::instance().run_command(
                                    commandBuffer_,
                                    /*async = */ false
    );
    return reinterpret_cast<int64_t*>(out_buffer.get_memory())[0];
}

template<typename T>
inline DeviceMTLShared ptr_to_shared_buffer(T* ptr, uint64_t size){
    DeviceMTLShared shared;
    shared.allocate_memory(size * sizeof(T));
    shared.adjust_type_bytes(sizeof(T));
    std::memcpy(shared.get_memory(), ptr, ptr + size);
}

template<typename T>
inline intrusive_ptr<DeviceMTLPrivate> ptr_to_private_buffer(T* ptr, uint64_t size){
    DeviceMTLShared staging_in = ptr_to_shared_buffer(ptr, size);
    DeviceMTLPrivate outBuffer;
    outBuffer.allocate_memory(sizeof(T) * size);
    shared.adjust_type_bytes(sizeof(T));
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = mtl::abs::MetalContext::instance().makeCommandBuffer();
    MTL::CommandBuffer* buffer = commandBuffer_->cmd;
    MTL::BlitCommandEncoder* blit = buffer->blitCommandEncoder();
    blit->copyFromBuffer(staging_in.get_buffer(), 0, outBuffer.get_buffer(), 0, sizeof(T) * size);
    blit->endEncoding();
    mtl::abs::MetalContext::instance().run_command(
                                    commandBuffer_,
                                    /*async = */ false
    );
    return make_intrusive<DeviceMTLPrivate>(outBuffer);
}

inline bool is_contiguous(
        const int64_t* shape,
        const int64_t* strides,
        const uint8_t dims) noexcept {
    int64_t expected = 1;
    int32_t i = int32_t(dims) - 1;
    for(; i >= 0; --i){
        if(shape[i] != 1 && strides[i] != expected)
            return false;
        expected *= shape[i];
    }
    return true;
}



// for concatenated, it is basically going to make a single indexed buffer
// and then change the offset according to that view's size and then delete the views no longer needed
BucketMTL BucketMTL::new_bounds_concatenated_(int64_t* ptr, int64_t size) const {
    intrusive_ptr<DeviceMTLPrivate> idx_device = ptr_to_private_buffer<int64_t>(ptr, size);
    intrusive_ptr<mtl::abs::MetalBuffer> buffer = idx_device->get_buffer();
    const auto& s = std::get<3>(this->storage);
    std::size_t dtype_size = DTypeFuncs::size_of_dtype(this->dtype_);
    int64_t mutable_size = size;
    size_t index = 0;
    while(mutable_size > 0){
        ++index;
        mutable_size -= (s.buffers[index].numelBytes / s.buffers[index].buffer->typeBytes);
    }
    std::vector<mtl::abs::MetalBufferView> new_views(index,
                            MetalBufferView{
                                .buffer = nullptr, // needs
                                .offsetBytes = 0,
                                .numelBytes = 0,   // needs
                                .idxOffset = 0,    // needs
                                .ndim = 0,
                                .sizes = nullptr,
                                .strides = nullptr,
                                .indexes = idx_device;
                            }
    );
    
    int64_t cur_idx_offset = 0;
    for(size_t i =0; i < index; ++i){
        new_views[i].buffer = s.buffers[i].buffer;
        new_views[i].numelBytes = s.buffers[i].numelBytes;
        new_views[i].idxOffset = cur_idx_offset;
        cur_idx_offset += (new_views[i].numelBytes / dtype_size);
    }
    // so when encoding on a concatenated tensor
    // The encoder will automatically increase the offset based on the current offset
    mtl::abs::MetalContext& ctx = mtl::abs::MetalContext::instance();
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer = ctx.makeCommandBuffer();

    mtl::abs::encodeCommand(
            mtl::abs::EncoderOptions{
                .commandBuffer = commandBuffer,
                .pipeline = nullptr,
                .size = s.total_numel,
                .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
            },
        makeEncoderArguments( // contiguous arguments
            ctx.get_pipeline(
                    "indexes_convert_contiguous_kernel" 
            ), // got pipeline
            mtl::abs::utils::EncodeBuffer{buffer, 0},
            mtl::abs::utils::ViewBufferArg::Offset,
            mtl::abs::utils::ViewBufferArg::TotalPN
        ),// done
        makeEncoderArguments(
            ctx.get_pipeline(
                    "indexes_convert_affine_kernel"
            ),
            mtl::abs::utils::EncodeBuffer{buffer, 0},
            mtl::abs::utils::ViewBufferArg::Offset,
            mtl::abs::utils::ViewBufferArg::Ndim,
            mtl::abs::utils::ViewBufferArg::Sizes,
            mtl::abs::utils::ViewBufferArg::Strides,
            mtl::abs::utils::ViewBufferArg::Numel,
            mtl::abs::utils::ViewBufferArg::TotalPN // minus
        ),
        makeEncoderArguments(
            ctx.get_pipeline(
                    "indexes_convert_strided_kernel"
            ),
            mtl::abs::utils::EncodeBuffer{buffer, 0},
            mtl::abs::utils::ViewBufferArg::Indexes,
            mtl::abs::utils::ViewBufferArg::Nume,
            mtl::abs::utils::ViewBufferArg::TotalPN // minus
        ), utils::span<mtl::abs::MetalBufferView>(s.buffers, index) // limits encoder's looking at each value it is meant to
    );

    return BucketMTL(
        BucketMTLConcatenated{
            .devices = s.devices,
            .buffers = std::move(new_views),
            .total_numel = size
        }, this->dtype_
    );

}

BucketMTL BucketMTL::new_bounds_mtl(int64_t* ptr, int64_t size, bool fix) const {
    utils::throw_exception(!this->is_null(), "Cannot get new bounds of null memory");
    if(this->is_concatenated()){
        return this->new_bounds_concatenated_(ptr, size);
    }
    int64_t offset = this->offset(/*strided = */false);
    int64_t* max = std::max_element(ptr, ptr + size);
    utils::throw_exception(
            *max < this->numel(),
            "Error, max element $ is out of range when creating new strides",
            *max);

    BucketMTL out_bucket(
            BucketMTLStrides{
                this->get_device(),
                ptr_to_private_buffer<int64_t>(ptr, size),
                0, // idx_offset
                size
            }, this->dtype_
    );

    if(((this->is_contiguous() || this->is_affine())
            && offset == 0) || !fix){
        return std::move(out_bucket);
    }
    
    //loading the shader library
    mtl::abs::MetalContext& ctx = mtl::abs::MetalContext::instance();
    const char** names = {"contiguous_kernel", "affine_kernel", "strided_kernel"};
    std::string kernelName = std::string("indexes_convert_")
                            + std::string(names[storage.index()]);

    intrusive_ptr<mtl::abs::Pipeline> pipeline_ = ctx.get_pipeline(kernelName);
    intrusive_ptr<mtl::abs::MetalBuffer> out_indexes = std::get<2>(out_bucket.storage).indexes->get_buffer(); 
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = ctx.makeCommandBuffer();
    if(this->is_strided()){
        const auto& s = std::get<2>(this->storage);
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = size,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{out_indexes->get_buffer(), 0},
                mtl::abs::utils::EncodeBuffer{s.indexes->get_buffer(), s.idx_offset * sizeof(int64_t)},
                nnz,
                static_cast<int64_t>(0) // minus
        );
    }
    else if(this->is_affine()){
        const auto& a = std::get<1>(this->storage);
        int64_t numel = this->numel();

        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = size,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{out_indexes->get_buffer(), 0},
                static_cast<int64_t>(a.offset),
                static_cast<uint32_t>(ndim),
                a.intrusive_sizes,
                a.intrusuve_strides,
                numel,
                static_cast<int64_t>(0) // minus 
        );
    }
    else if(this->is_contiguous){ // is contiguous
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = size,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{out_indexes->get_buffer, 0}
                offset,
                static_cast<int64_t>(0) // minus 
        );
    }
    
    ctx.run_command(commandBuffer_, /* async = */ false);
    return std::move(out_bucket);  
}

// NOTE: at a higher level API (The nt::Tensor level), this should go into account about errors involving strides on the GPU
//  This allows it to work because the whole point of the Bucket is just working, but actual user use should handle that
//  This note will go away once handled
// for concatenated, it just contigutized and then made the new bounds
BucketMTL BucketMTL::new_bounds_mtl(int64_t offset, utils::span<int64_t> sizes_, utils::span<int64_t> strides_) const {
    utils::throw_exception(!this->is_null(), "Cannot get new bounds of null memory");
    utils::throw_exception(sizes_.size() == strides_.size(),
            "Error, expected sizes and strides to have the same size for the new number of dimensions");
    const uint8_t ndim = sizes_.size();
    const int64_t* sizes = sizes_.data();
    const int64_t* strides = strides_.data();
    if(this->is_strided() || this->is_concatenated()){
        return this->contiguous_mtl().new_bounds_mtl(offset, sizes_, strides_);
    }
    int64_t cur_numel = this->numel();
    int64_t n_numel = 1;
    for(uint8_t i = 0; i < ndim)
        n_numel *= sizes[i];
    utils::throw_exception((n_numel + offset) <= this->numel(),
            "Error: Expected new numel ($) to be less than or equal to current numel ($)",
            n_numel, cur_numel);
    int64_t cur_offset = this->offset(false);
    if(is_contiguous(sizes, strides, ndim)){
        return BucketMTL(
                BucketMTLContiguous{
                this->get_device(),
                offset + cur_offset, n_numel});

    }
    intrusive_ptr<intrusive_tracked_list_sub<int64_t, false>> intrusive_sizes_ = make_intrusive<intrusive_tracked_list_sub<int64_t, false>>(ndim);
    intrusive_ptr<intrusive_tracked_list_sub<int64_t, false>> intrusive_strides_ = make_intrusive<intrusive_tracked_list_sub<int64_t, false>>(ndim);
    int64_t* sizes_heap = intrusive_sizes_->get();
    int64_t* strides_heap = intrusive_strides_->get();
    for(uint8_t i = 0; i < ndim; ++i){
        strides_heap[i] = strides[i];
        sizes_heap[i] = sizes[i];
    }
    return BucketMTL(
            BucketMTLAffine{
                this->get_device(),
                offset + cur_offset,
                ndim,
                intrusive_sizes_, intrusive_strides_
            }, this->dtype_
    );
}


// TODO: This needs to be validated and ran tests on

inline int64_t determine_new_offset(
        int64_t idx,
        int64_t in_offset, uint8_t ndim, 
        int64_t* sizes, int64_t* strides){
    for (int d = ndim - 1; d >= 0 && idx > 0; --d) {
        uint64_t coord = idx % sizes[d];
        idx /= sizes[d];
        in_offset += coord * strides[d];
    }
    return in_offset;
}

inline bool flat_index_to_indices_strided(
    int64_t flat_index,
    const int64_t* sizes,
    const int64_t* strides,
    uint8_t ndim,
    int64_t* out_indices
) noexcept
{
    if (flat_index < 0)
        return false;

    // Compute max possible flat_index
    int64_t max_index = 0;
    for (uint8_t i = 0; i < ndim; ++i) {
        if (sizes[i] <= 0)
            return false;
        max_index += (sizes[i] - 1) * strides[i];
    }

    if (flat_index > max_index)
        return false;

    // Decode indices according to strides
    for (uint8_t i = 0; i < ndim; ++i) {
        if (strides[i] == 0) {
            out_indices[i] = 0; // broadcasting dimension
            continue;
        }
        out_indices[i] = flat_index / strides[i];
        flat_index %= strides[i];
    }

    return true;
}

// using this when I don't want strides taken into account
inline bool flat_index_to_indices(
    int64_t flat_index,
    const int64_t* sizes,
    uint8_t ndim,
    int64_t* out_indices
) noexcept
{
    if (flat_index < 0)
        return false;
    for (int d = ndim-1; d >= 0; --d) {
        out_indices[d] = flat_index % sizes[d];
        flat_index /= sizes[d];
    }
    return flat_index == 0;
}

#define NT_VIEW_NUMEL(view) (view.numelBytes / static_cast<int64_t>(view.buffer->typeBytes))
BucketMTL BucketMTL::new_bounds_concatenated_(int64_t absolute_start, int64_t absolute_end) const {
    const auto& s = std::get<3>(storage);
    const int64_t n_total_numel = end - start;
    const std::vector<mtl::abs::MetalBufferView>& views = s.buffers;
    size_t dtype_size = this->byte_size();
    int64_t relative_start = absolute_start;
    int64_t relative_end = absolute_end;
    size_t start_index = 0;
    while(relative_start > 0 && start_index < views.size()){
        int64_t numel = NT_VIEW_NUMEL(views[start_index]);
        if(relative_start < numel)
            break;
        if(relative_start == numel){
            ++start_index;
            relative_start = 0;
            break;
        }
        relative_start -= numel;
        end -= numel;
        ++start_index;
    }
    if(start_index >= views.size() || NT_VIEW_NUMEL(views[start_index]) <= relative_start){
        utils::THROW_EXCEPTION(false,
                "Error, views size and start index do not correspond");
    }

    size_t end_index = start_index;
    while(relative_end > 0 && end_index < views.size()){
        int64_t numel = NT_VIEW_NUMEL(views[end_index]);
        if(relative_end < numel){
            break;
        }
        if(relative_end == numel){
            relative_end = 0; // marks that it's the whole thing
        }
        relative_end -= numel;
        ++end_index;
    }
    // handles if no longer concatenated view:
    if(start_index == end_index){
        return BucketMTL(views[start_index], s.devices.get(start_index), this->dtype_);
    }
    // adjusting the view buffers
    std::vector<mtl::abs::MetalViewBuffer> nviews((end_index - start_index + 1), 
                                    mtl::abs::MetalViewBuffer{
                                            .buffer = nullptr,
                                            .offsetBytes = 0,
                                            .numelBytes = 0,
                                            .idxOffset = 0,
                                            .ndim = 0,
                                            .sizes = nullptr,
                                            .strides = nullptr,
                                            .indexes = nullptr
                                    }
    );
    
    nviews[0] = views[start_index];
    if(relative_start != 0){
        if(nviews[0].indexes != nullptr){ // strided
            nviews[0].idxOffset += relative_start;
        }else if(nviews[0].sizes != nullptr && nviews[0].strides != nullptr){ // affine
            nviews[0] = (BucketMTL(nviews[0], s.devices->get(start_index), this->dtype_) + relative_start).get_metal_buffer_view();
        }
        else{ // contiguous
            nviews[0].offsetBytes += (relative_start * dtype_size);
        }
    }

    nviews.back() = views[end_index];
    if(relative_end != 0){
        if(nviews.back().sizes == nullptr && nviews.back().strides == nullptr){ // strided or contiguous
            nviews.back().numelBytes = (relative_end * dtype_size);
        }else if(nviews.back().sizes != nullptr){ // affine
            nviews.back() = (BucketMTL(nviews.back(), s.devices->get(end_index), this->dtype_).new_bounds_mtl(0, relative_end)).get_metal_buffer_view();
        }
    }
    if((end_index - start_index) > 1)
        std::copy(views.cbegin() + (start_index + 1), views.cbegin() + (end_index), nviews.begin() + 1);
    intrusive_ptr<DeviceHolder> out_holder = make_intrusive<DeviceHolder>(end_index - start_index);
    std::copy(&s.devices.get(start_index), &s.devices.get(end_index), &out_holder.get(0));
    return BucketMTL(
            BucketMTLConcatenated{
                .devices = out_holder,
                .buffers = std::move(nviews),
                .total_numel = n_total_numel
            }, this->dtype_);
}
#undef NT_VIEW_NUMEL


BucketMTL BucketMTL::new_bounds_mtl(int64_t start, int64_t end) const {
    utils::throw_exception(!this->is_null(), "Cannot get new bounds of null memory");
    int64_t offset = this->offset(false);
    utils::throw_exception(end > start,
            "Error, expected end ($) to be greater than start ($)"
            " for new bounds", end, start);
    
    utils::throw_exception(end <= this->numel(),
            "Error, expected end ($) to be less than or equal to current numel ($)",
            (end), this->numel());
    utils::throw_exception(start <= this->numel(),
            "Error, expected start ($) to be less than or equal to current numel ($)",
            start, this->numel());

    if(this->is_contiguous()){
        int64_t n_numel = end - start;
        start += offset;
        return BucketMTL(
                BucketMTLContiguous{this->get_device(), start, n_numel},
                this->dtype_
        );
    }
    else if(this->is_strided()){
        const auto& s = std::get<2>(this->storage);
        return BucketMTL(
            BucketMTLStrided{
                .storage_ = s.storage_,
                .indexes = s.indexes,
                .idx_offset = s.idx_offset + start,
                .nnz = (end - start)
            }, this->dtype_
        )
    }
    // TODO: This needs to have general tests ran on it, not entirely convinced it is correct
    else if(this->is_affine()){
        // so here, we are going to figure out what the new strides and size is
        // based off the new bounds
        int64_t numel = this->numel();
        const auto& a = std::get<1>(storage);
        fallback_strided:
        {
            return this->bucket_all_indices_mtl().new_bounds_mtl(start, end);
        }
        if (is_contiguous(a.sizes(), a.strides(), a.ndim)) {
            int64_t base = this->offset(false);
            return BucketMTL(
                BucketMTLContiguous{
                    this->get_device(),
                    base + start,
                    (end - start)
                },
                this->dtype_
            );
        }

        const int64_t* sizes = a.sizes();
        const int64_t* strides = a.strides();
        const uint8_t cur_dims = a.ndim;
        int64_t n_numel = end - start;

        if(cur_dims == 1 || n_numel < sizes[cur_dims-1]){
            int64_t n_offset = 
                determine_new_offset(start, a.offset, 
                                        cur_dims, sizes, 
                                        strides);
             return this->new_bounds_mtl(n_offset, 1, &n_numel, &strides[cur_dims-1]);
        }


        // first thing to check is if (start % sizes[cur_dims-1] == 0);
        if(!(start % sizes[cur_dims-1] == 0)){
            goto fallback_strided;
        }

        NT_VLA(int64_t, out_indices_start, cur_dims);
        NT_VLA(int64_t, out_indices_end, cur_dims);
        bool start_indicable = flat_index_to_indices(
            start, sizes, cur_dims, out_indices_start
            // int64_t flat_index,
            // const int64_t* sizes,
            // uint8_t ndim,
            // int64_t* out_indices
        );
        bool end_indicable = flat_index_to_indices(
            end-1, sizes, cur_dims, out_indices_end
        );
        if(!start_indicable || !end_indicable){
            NT_VLA_DEALC(out_indices_start);
            NT_VLA_DEALC(out_indices_end);
            goto fallback_strided;
        }


        int32_t cut_dim = -1;
        for (uint32_t i = 0; i < cur_dims; ++i) {
            if (out_indices_start[i] != out_indices_end[i]) {
                cut_dim = i;
                break;
            }
        }

        if(cut_dim == -1 || out_indices_end[cut_dim] <= out_indices_start[cut_dim]){ 
            // should be impossible
            // would be 1-element tensor
            // which was alread covered
            NT_VLA_DEALC(out_indices_start);
            NT_VLA_DEALC(out_indices_end);
            goto fallback_strided;
        }

        // validating remaining dimensions
        for (uint32_t i = cut_dim + 1; i < cur_dims; ++i) {
            if (out_indices_start[i] != 0 ||
                out_indices_end[i] != (sizes[i] - 1)) {
                // if this is true cannot be affine
                NT_VLA_DEALC(out_indices_start);
                NT_VLA_DEALC(out_indices_end);
                goto fallback_strided;
            }
        }

        uint8_t new_dims = cut_dim + 1;
        NT_VLA(int64_t, new_sizes, new_dims);
        NT_VLA(int64_t, new_strides, new_dims);

        for (uint32_t i = 0; i < cut_dim; ++i) {
            new_sizes[i]   = sizes[i];
            new_strides[i] = strides[i];
        }

        new_sizes[cut_dim] =
            out_indices_end[cut_dim] - out_indices_start[cut_dim] + 1;
        new_strides[cut_dim] = strides[cut_dim];
        
        int64_t new_offset = determine_new_offset(
            start,
            a.offset,
            cur_dims,
            sizes,
            strides
        );

        BucketMTL out_bucket = this->new_bounds_mtl(
                new_offset, new_dims, new_sizes, new_strides
        );

        NT_VLA_DEALC(out_indices_start);
        NT_VLA_DEALC(out_indices_end);
        NT_VLA_DEALC(new_sizes);
        NT_VLA_DEALC(new_strides);
        return std::move(out_bucket);
    }
    // else is concatenated
    return this->new_bounds_concatenated_(start, end);
}

BucketMTL BucketMTL::operator+(int64_t i) const {
    return this->new_bounds_mtl(i, this->numel());
}

BucketMTL BucketMTL::operator-(int64_t i) const {
    return this->new_bounds_mtl(0, this->numel() - i);
}

BucketMTL BucketMTL::bucket_all_indices_concatenated() const {
    // first going to check if all the strides are already indexed in the concatenated view
    const auto& s = std::get<3>(this->storage);
    const intrusive_ptr<DeviceMTLPrivate>& idx_check = s.buffers[0].indexes;
    if(idx_check != nullptr){
        bool same = true;
        for(const auto& view : s.views){
            if(idx_check != view.indexes || view.indexes == nullptr){
                same = false;
                return
            }
        }
        if(same) return *this;
    }
    // at this point I know all the strides are not explicitly indexed
    int64_t numel = this->numel();
    size_t dtype_size = DTypeFuncs::size_of_dtype(this->dtype_);

    intrusive_ptr<DeviceMTLPrivate> strides = make_intrusive<DeviceMTLPrivate>();
    strides->allocate_memory(DType::int64, numel);
    intrusive_ptr<mtl::abs::MetalBuffer> buffer = strides->get_buffer();
    mtl::abs::MetalContext ctx = mtl::abs::MetalContext::instance();
    intrusive_ptr<mtl::abs::Pipeline> iota_pipeline_ = ctx.get_pipeline("iota_contiguous_kernel_long");
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer = ctx.makeCommandBuffer();
    mtl::abs::encodeCommand(
            mtl::abs::EncoderOptions{
                .commandBuffer = commandBuffer,
                .pipeline = iota_pipeline,
                .size = numel,
                .type_bytes = dtype_size
            },
            mtl::abs::utils::EncodeBuffer{buffer, 0},
            static_cast<int64_t>(0) // start
    );

    mtl::abs::encodeCommand(
            mtl::abs::EncoderOptions{
                .commandBuffer = commandBuffer,
                .pipeline = nullptr,
                .size = s.total_numel,
                .type_bytes = dtype_size
            },
        makeEncoderArguments( // contiguous arguments
            ctx.get_pipeline(
                    "indexes_convert_contiguous_kernel" 
            ), // got pipeline
            mtl::abs::utils::EncodeBuffer{buffer, 0},
            mtl::abs::utils::ViewBufferArg::Offset,
            mtl::abs::utils::ViewBufferArg::TotalPN
        ),// done
        makeEncoderArguments(
            ctx.get_pipeline(
                    "indexes_convert_affine_kernel"
            ),
            mtl::abs::utils::EncodeBuffer{buffer, 0},
            mtl::abs::utils::ViewBufferArg::Offset,
            mtl::abs::utils::ViewBufferArg::Ndim,
            mtl::abs::utils::ViewBufferArg::Sizes,
            mtl::abs::utils::ViewBufferArg::Strides,
            mtl::abs::utils::ViewBufferArg::Numel,
            mtl::abs::utils::ViewBufferArg::TotalPN // minus
        ),
        makeEncoderArguments(
            ctx.get_pipeline(
                    "indexes_convert_strided_kernel"
            ),
            mtl::abs::utils::EncodeBuffer{buffer, 0},
            mtl::abs::utils::ViewBufferArg::Indexes,
            mtl::abs::utils::ViewBufferArg::Nume,
            mtl::abs::utils::ViewBufferArg::TotalPN // minus
        ), s.buffers
    );

    std::vector<MetalBufferView> new_views(s.buffers.size(),
                            MetalBufferView{
                                .buffer = nullptr, // needs
                                .offsetBytes = 0,
                                .numelBytes = 0,   // needs
                                .idxOffset = 0,    // needs
                                .ndim = 0,
                                .sizes = nullptr,
                                .strides = nullptr,
                                .indexes = strides;
                            }
    );

    int64_t cur_idx_offset = 0;
    for(size_t i =0; i < s.buffers.size(); ++i){
        new_views[i].buffer = s.buffers[i].buffer;
        new_views[i].numelBytes = s.buffers[i].numelBytes;
        new_views[i].idxOffset = cur_idx_offset;
        cur_idx_offset += (new_views[i].numelBytes / dtype_size);
    }
    
    return BucketMTL(
        BucketMTLConcatenated{
            .devices = s.devices,
            .buffers = std::move(new_views),
            .total_numel = s.total_numel
        }, this->dtype_
    );

}

BucketMTL BucketMTL::bucket_all_indices_mtl() const {
    utils::throw_exception(!this->is_null(), "Cannot bucket all indices of null memory");
    if(this->is_strided())
        return *this;
    if(this->is_concatenated())
        return this->bucket_all_indices_concatenated();
    int64_t numel = this->numel();
    if(this->is_contiguous()){
        // specific faster path
        intrusive_ptr<DeviceMTLPrivate> strides = make_intrusive<DeviceMTLPrivate>();
        strides->allocate_memory(DType::int64, numel);
        mtl::iota(strides, numel, std::get<0>(storage).offset);
        return BucketMTL(
                BucketMTLStrided({this->get_device(), strides, 0, numel})
        );
    }
    // affine
    intrusive_ptr<DeviceMTLPrivate> strides = make_intrusive<DeviceMTLPrivate>();
    strides->allocate_memory(DType::int64, numel);
    mtl::abs::MetalContext ctx = mtl::abs::MetalContext::instance();

    intrusive_ptr<mtl::abs::Pipeline> iota_pipeline_ = ctx.get_pipeline("iota_contiguous_kernel_long");
    intrusive_ptr<mtl::abs::Pipeline> fix_pipeline_ = ctx.get_pipeline("indexes_convert_affine_kernel"); 
    MTL::ComputePipelineState* iota_pipeline = iota_pipeline_->pipeline();
    MTL::ComputePipelineState* fix_pipeline = fix_pipeline_->pipeline();

    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = ctx.makeCommandBuffer();
    mtl::abs::encodeCommand(
            mtl::abs::EncoderOptions{
                .commandBuffer = commandBuffer_,
                .pipeline = iota_pipeline,
                .size = numel,
                .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
            },
            mtl::abs::utils::EncodeBuffer{strides->get_buffer(), 0},
            static_cast<int64_t>(0) // start
    );
    const auto& a = std::get<1>(this->storage);
    uint32_t ndim = a.ndim;
    mtl::abs::encodeCommand(
            mtl::abs::EncoderOptions{
                .commandBuffer = commandBuffer_,
                .pipeline = fix_pipeline,
                .size = numel,
                .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
            },
            mtl::abs::utils::EncodeBuffer{strides->get_buffer(), 0},
            a.offset,
            static_cast<uint32_t>(ndim),
            a.intrusive_sizes,
            a.intrusive_strides,
            numel
    );


    ctx.run_command(commandBuffer_);

    return BucketMTL(
            BucketMTLStrided{this->get_device(), strides, 0, numel},
            this->dtype_
    );
}


template<>
std::vector<BucketMTL> BucketMTL::split_contiguous_<std::vector<BucketMTL>>(uint64_t splitting) const {
    int64_t numel = this->numel();
    if(splitting > numel){
        return std::vector<BucketMTL> {*this};
    }
    
    uint64_t div = numel / splitting;
    uint64_t remainder = numel % splitting;
    bool r = remainder > 0;
    
    auto& s = std::get<0>(this->storage);
    intrusive_ptr<Device> dev = s.storage();
    int64_t current = s.offset;
	std::vector<BucketMTL> fb( (r) ? div + 1 : div, 
                                BucketMTL(
                                    BucketMTLContiguous{dev, current, splitting},
                                    this->dtype_;
                                )
                            ); 

    for(uint64_t i = 0; i < div; ++i){
        std::get<0>(fb[i].storage).offset = current;
        current += splitting;
    }
    if(r){
        std::get<0>(fb.back().storage).offset = current;
        std::get<0>(fb.back().storage).numel = remainder;
    }
    return std::move(fb);
}


// could figure out how to split it properly and intertwined for every single type of inner-view
// however, that will be pretty bug-prone and it would be more important to work on other things
template<>
std::vector<BucketMTL> BucketMTL::split_concatenated_<std::vector<BucketMTL>>(uint64_t splitting) const {
    // just going to increase the idx_offset every time basically
    int64_t numel = this->numel();
    if(splitting > numel){
        return std::vector<BucketMTL> {*this};
    }

    BucketMTL strided = this->bucket_all_indices_mtl();
    
    
    uint64_t div = numel / splitting;
    uint64_t remainder = numel % splitting;
    bool r = remainder > 0;
    int64_t current = 0;
    std::vector<BucketMTL> fb;
    fb.reserve((r) ? div + 1 : div);
    for(uint64_t i = 0; i < div; ++i){
        fb.emplace_back(strided.new_bounds_mtl(current, current + splitting));
        current += splitting;
    }
    if(r){
        fb.emplace_back(strided.new_bounds_mtl(current, numel));
    }
    return std::move(fb);
}

template<>
std::vector<BucketMTL> BucketMTL::split_strided_<std::vector<BucketMTL>>(uint64_t splitting) const {
    // just going to increase the idx_offset every time basically
    int64_t numel = this->numel();
    if(splitting > numel){
        return std::vector<BucketMTL> {*this};
    }
    
    uint64_t div = numel / splitting;
    uint64_t remainder = numel % splitting;
    bool r = remainder > 0;
    
    auto& s = std::get<2>(this->storage);
    intrusive_ptr<Device> dev = s.storage;
    intrusive_ptr<DeviceMTLPrivate> indexes
    int64_t current = s.idx_offset;


	std::vector<BucketMTL> fb( (r) ? div + 1 : div, BucketMTL(
                                    BucketMTLStrided{
                                        dev, indexes, current, splitting
                                    }, this->dtype_
                )); 


    for(uint64_t i = 0; i < div; ++i){
        std::get<2>(fb[i].storage).idx_offset = current;
        current += splitting;
    }
    if(r){
        std::get<2>(fb.back().storage).idx_offset = current;
        std::get<2>(fb.back().storage).nnz = remainder;
    }
    return std::move(fb);
}



template<>
std::vector<BucketMTL> BucketMTL::split_affine_<std::vector<BucketMTL>>(uint64_t splitting) const {
    // just going to increase the idx_offset every time basically
    
    int64_t numel = this->numel();
    if(splitting > numel){
        return std::vector<BucketMTL> {*this};
    }
    fallback_strided:
    {
        return this->bucket_all_indices_mtl().split_strided_<std::vector<BucketMTL>>(splitting);
    }

    
    uint64_t div = numel / splitting;
    if(numel % splitting != 0)
        goto fallback_strided;

    auto& s = std::get<1>(this->storage);
    const int64_t* sizes = s.sizes();
    const int64_t* strides = s.strides();
    // splitting must be a combination of (the bottom) sizes in order for this to work
    int32_t dims = static_cast<int32_t>(s.ndim);
    // determining each offset:
    // offset = determine_new_offset(i * div, std::get<1>(fb[i-1].storage).offset,
    //                              dims, sizes, strides);
    
    intrusive_ptr<intrusive_tracked_list_sub<int64_t, false>> intrusive_sizes(nullptr);
    intrusive_ptr<intrusive_tracked_list_sub<int64_t, false>> intrusive_strides(nullptr);
    uint8_t ndim_size;
    if(splitting <= sizes[dim-1]){
        ndim_size = 1;
        intrusive_sizes = make_intrusive<intrusive_tracked_list_sub<int64_t, false>>(ndim_size);
        intrusive_strides = make_intrusive<intrusive_tracked_list_sub<int64_t, false>>(ndim_size);
        intrusive_sizes->get()[0] = splitting;
        intrusive_strides->get()[0] = strides[dims-1];
    }else{
        ndim_size = 1;
        uict64_t cpy = splitting;
        if(cpy % sizes[dim-1] != 0) goto fallback_strided;
        cpy /= sizes[dim-1];
        ++ndim_size;
        for(int32_t i = dims-2; i >= 0; --i){
            if(cpy < sizes[i] || cpy == 1){break;}
            if(cpy % sizes[i] != 0) goto fallback_strided;
            cpy /= sizes[i];
            ++ndim_size;
        }
        if(cpy == 1) --ndim_size;
        intrusive_sizes = make_intrusive<intrusive_tracked_list_sub<int64_t, false>>(ndim_size);
        intrusive_strides = make_intrusive<intrusive_tracked_list_sub<int64_t, false>>(ndim_size);
        std::memcpy(intrusive_sizes->get(), sizes + (dims - ndim_size), sizeof(int64_t) * ndim_size);
        std::memcpy(intrusive_strides->get(), strides + (dims - ndim_size), sizeof(int64_t) * ndim_size);
        if(cpy != 1){
            intrusive_sizes->get()[0] = cpy;
        }
    }

    intrusive_ptr<DeviceHolder> dev = s.storage_;
    int64_t current = s.offset;

    std::vector<BucketMTL> fb(div, BucketMTL(
                                    BucketMTLAffine{
                                    .storage_ = dev,
                                    .ndim = ndim_size,
                                    .offset = current,
                                    .intrusive_sizes = intrusive_sizes,
                                    .intrusive_strides = intrusive_strides
                                    }, this->dtype_));
    
    int64_t idx = splitting;
    for(uint64_t i = 1; i < div; ++i, idx += splitting){
        current = determine_new_offset(idx, current,
                                ndim_size, sizes, strides);
        std::get<1>(fb[i].storage).offset = current;
    }

    return std::move(fb);
}

// need to make a split concatenated

template<>
std::vector<BucketMTL> BucketMTL::split<std::vector<BucketMTL>>(uint64_t splitting) const {
    utils::throw_exception(!this->is_null(), "Cannot split null memory");
    if(this->is_contiguous()){
        return this->split_contiguous_<std::vector<BucketMTL>>(splitting);
    }else if(this->is_strided()){
        return this->split_strided_<std::vector<BucketMTL>>(splitting);
    }else if(this->is_affine()){
        return this->split_affine_<std::vector<BucketMTL>>(splitting);
    }else if(this->is_concatenated()){
        return this->split_concatenated_<std::vector<BucketMTL>>(splitting);
    }
}

//TODO: Once a bucket generalization for the Tensor class is created to make these viable, fix this function
// which takes a general bucket and makes it a mtl bucket
inline void mtl_bkt_insert(Bucket& bkt, BucketMTL& mtl_bkt) noexcept {return;}

template<>
Tensor BucketMTL::split_contiguous_<Tensor>(uint64_t splitting) const {
    int64_t numel = this->numel();
    if(splitting > numel){
        Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
        mtl_bkt_insert(t._vals.bucket, *this);
		t._vals.size = numel;
		t._total_size = numel;
		t._size = SizeRef({t._total_size});
		return output;
    }

    
    uint64_t div = numel / splitting;
    uint64_t remainder = numel % splitting;
    bool r = remainder > 0;
   

    auto& s = std::get<0>(this->storage);
    intrusive_ptr<DeviceHolder> dev = s.storage_;
    int64_t current = s.offset;

    BucketMTL cpy_bkt(
        BucketMTLContiguous{dev, current, splitting},
        this->dtype_;
    );


    Tensor output = Tensor::makeNullTensorArray((r) ? div + 1 : div);
    Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(output.data_ptr_end());


    for(uint64_t i = 0; i < div; ++i, ++begin){
        mtl_bkt_insert(begin->_vals.bucket, cpy_bkt);
        std::get<0>(cpy_bkt.storage).offset = current;
        current += splitting;
		begin->_vals.size = splitting;
		begin->_total_size = splitting;
		begin->_size = SizeRef({begin->_total_size});
    }
    if(r){
        std::get<0>(cpy_bkt.storage).offset = current;
        std::get<0>(cpy_bkt.storage).numel = remainder;
        mtl_bkt_insert(begin->_vals.bucket, cpy_bkt);
		begin->_vals.size = remainder;
		begin->_total_size = remainder;
		begin->_size = SizeRef({begin->_total_size});
    }
    return std::move(output);
}

template<>
Tensor BucketMTL::split_strided_<Tensor>(uint64_t splitting) const {
    int64_t numel = this->numel();
    if(splitting > numel){
        Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
        mtl_bkt_insert(t._vals.bucket, *this);
		t._vals.size = numel;
		t._total_size = numel;
		t._size = SizeRef({t._total_size});
		return output;
    }

    
    uint64_t div = numel / splitting;
    uint64_t remainder = numel % splitting;
    bool r = remainder > 0;
    
    auto& s = std::get<2>(this->storage);
    intrusive_ptr<DeviceHolder> dev = s.storage_;
    intrusive_ptr<DeviceMTLPrivate> indexes
    int64_t current = s.idx_offset;    
    Tensor output = Tensor::makeNullTensorArray((r) ? div + 1 : div);
    Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(output.data_ptr_end());

    BucketMTL cpy_bkt(
            BucketMTLStrided{
                dev, indexes, current, splitting
            }, this->dtype_
    ); 

    for(uint64_t i = 0; i < div; ++i, ++begin){
        std::get<2>(cpy_bkt.storage).idx_offset = current;
        mtl_bkt_insert(begin->_vals.bucket, cpy_bkt);
        current += splitting;
		begin->_vals.size = splitting;
		begin->_total_size = splitting;
		begin->_size = SizeRef({splitting});
    }
    if(r){
        std::get<2>(cpy_bkt.storage).offset = current;
        std::get<2>(cpy_bkt.storage).nnz = remainder;
        mtl_bkt_insert(begin->_vals.bucket, cpy_bkt);
		begin->_vals.size = remainder;
		begin->_total_size = remainder;
		begin->_size = SizeRef({remainder});
    }
    return std::move(output);
}

template<>
Tensor BucketMTL::split_concatenated_<Tensor>(uint64_t splitting) const {
    int64_t numel = this->numel();
    if(splitting > numel){
        Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
        mtl_bkt_insert(t._vals.bucket, *this);
		t._vals.size = numel;
		t._total_size = numel;
		t._size = SizeRef({t._total_size});
		return output;
    }

    BucketMTL strided = this->bucket_all_indices_mtl();

    
    uint64_t div = numel / splitting;
    uint64_t remainder = numel % splitting;
    bool r = remainder > 0;

    int64_t current = 0;    
    Tensor output = Tensor::makeNullTensorArray((r) ? div + 1 : div);
    Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(output.data_ptr_end());


    for(uint64_t i = 0; i < div; ++i, ++begin){
        mtl_bkt_insert(begin->_vals.bucket, strided.new_bounds_mtl(current, current + splitting));
        current += splitting;
		begin->_vals.size = splitting;
		begin->_total_size = splitting;
		begin->_size = SizeRef({splitting});
    }
    if(r){
        mtl_bkt_insert(begin->_vals.bucket, strided.new_bounds_mtl(current, numel));
		begin->_vals.size = remainder;
		begin->_total_size = remainder;
		begin->_size = SizeRef({remainder});
    }
    return std::move(output);
}

template<>
Tensor BucketMTL::split_affine_<Tensor>(uint64_t splitting) const {
    int64_t numel = this->numel();

    if(splitting > numel){
        Tensor output = Tensor::makeNullTensorArray(1);
		Tensor& t = output.item<Tensor>();
        mtl_bkt_insert(t._vals.bucket, *this);
		t._vals.size = numel;
		t._total_size = numel;
		t._size = SizeRef(std::get<1>(this->storage).sizes(), static_cast<int64_t>(std::get<1>(this->storage).ndim));
		return output;
    }
    if(numel % splitting != 0)
        return this->bucket_all_indices_mtl().split_strided_<Tensor>(splitting);
    std::vector<BucketMTL> splits = this->split_affine_<std::vector<BucketMTL>>(splitting);

    SizeRef each_size = splits[0].is_affine() ? SizeRef(std::get<1>(splits[0].storage).sizes(), 
                                                                std::get<1>(splits[0].storage).ndim)
                                              : SizeRef({splitting});

    // BucketMTL cpy_bkt(
    //         BucketMTLAffine{
    //         .storage = dev,
    //         .ndim = ndim_size,
    //         .offset = current,
    //         .intrusive_sizes = intrusive_sizes,
    //         .intrusive_strides = intrusive_strides
    //         }, this->dtype_);

    Tensor output = Tensor::makeNullTensorArray(div);
    Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
    Tensor* end = reinterpret_cast<Tensor*>(output.data_ptr_end()); 

    for(uint64_t i = 0; i < div; ++i, ++begin){
        mtl_bkt_insert(begin->_vals.bucket, splits[i]);
		begin->_vals.size = splitting;
		begin->_total_size = splitting;
		begin->_size = each_size.clone();
    }
    
    return std::move(output);
}

template<>
Tensor BucketMTL::split<Tensor>(uint64_t splitting) const {
    utils::throw_exception(!this->is_null(), "Cannot split null memory");
    if(this->is_contiguous()){
        return this->split_contiguous_<Tensor>(splitting);
    }else if(this->is_strided()){
        return this->split_strided_<Tensor>(splitting);
    }else if (this->is_affine()){
        return this->split_affine_<Tensor>(splitting);
    }else if (this->is_concatenated()){
        return this->split_concatenated_<Tensor>(splitting);
    }
}


inline int64_t fix_ranges(const int64_t& numel, std::vector<std::pair<int64_t, int64_t>>& ranges) {
    int64_t total_size = 0;
    ranges[0].first = ranges[0].first < 0 ? ranges[0].first + numel : ranges[0].first;
    ranges[0].second = ranges[0].second < 0 ? ranges[0].second + numel : ranges[0].second;
    if(!(ranges[0].first < ranges[0].second && ranges[0].second <= numel)) return 0;
    utils::throw_exception(ranges[0].first < ranges[0].second && ranges[0].second <= numel,
            "Error, got invalid splitting range of {$, $} for tensor with $ elements", ranges[0].first, ranges[0].second, numel);
    total_size += ranges[0].second - ranges[0].first;
    for(size_t i = 1; i < ranges.size(); ++i){
        std::pair<int64_t, int64_t>& p = ranges[i];
        p.first = p.first < 0 ? p.first + numel : p.first;
        p.second = p.second < 0 ? p.second + numel : p.second;
        if(!(p.first < p.second && p.second <= numel)) return i; 
        total_size += p.second - p.first;
        p.second = total_size;
    }
    return -1;
}


inline int64_t fix_ranges_2(const int64_t& numel, std::vector<std::pair<int64_t, int64_t>>& ranges) {
    int64_t total_size = 0;
    for(auto& range = ranges.begin(); range != ranges.end(); ++range){
        range->first = range->first < 0 ? range->first + numel : range->first;
        range->second = range->second < 0 ? range->second + numel : range->second;
        utils::throw_exception(range->first < range->second && range->second <= numel,
            "Error, got invalid splitting range of {$, $} for tensor with $ elements", range->first, range->second, numel);
    }
    return -1;
}


// for concatenated it is going to do similar to the bucket_all_indices where
// it is going to just distribute the index as a singular device around
BucketMTL BucketMTL::catV(std::vector<BucketMTL> buckets){
    size_t indexes = 0;
    int64_t total_numel = 0;
    for(const auto& bkt : buckets){
        utils::throw_exception(!bkt.is_null(), "Error: cannot concatenate null memory");
        if(bkt.is_concatenated())
            indexes += std::get<3>(bkt.storage).buffers.size();
        else
            ++indexes;
        total_numel += bkt.numel();
    }
    std::vector<mtl::abs::MetalBufferView> views;
    views.reserve(indexes);
    intrusive_ptr<DeviceHolder> devices = make_intrusive<DeviceHolder>(indexes);
    int64_t index = 0;
    for(const auto& bkt : buckets){
        if(bkt.is_concatenated()){
            const auto& s = std::get<3>(bkt.storage);
            std::copy(&s.devices->get(0), &s.devices->get(s.buffers.size()), &devices->get(index));
        }else{
            devices->get(index) = bkt.get_device();
        }
        ++index;
        bkt.emplace_metal_buffer_view(views);
    }
    return BucketMTL(
            BucketMTLConcatenated{
                .devices = devices,
                .buffers = std::move(views),
                .total_numel = total_numel
            }, this->dtype_);
}
template<>
BucketMTL BucketMTL::range_concatenated_<BucketMTL>(std::vector<std::pair<int64_t, int64_t>> ranges) const {
    if(ranges.size() == 0)
        return BucketMTL(nullptr);
    int64_t numel = this->numel();
    fix_ranges_2(numel, ranges);
    // just going to increase the idx_offset every time basically
    int64_t numel = this->numel();


    BucketMTL strided = this->bucket_all_indices_mtl();
    
    std::vector<BucketMTL> fb;
    fb.reserve(ranges.size());
    for(const auto& range : ranges){
        fb.emplace_back(strided.new_bounds_mtl(range.first, range.second));
    }
    return BucketMTL::catV(std::move(fb));

}

template<>
BucketMTL BucketMTL::range<BucketMTL>(std::vector<std::pair<int64_t, int64_t>> ranges) const {
    utils::throw_exception(!this->is_null(), "Cannot range null memory");
    if(this->is_concatenated()){
        return this->range_concatenated_<BucketMTL>(std::move(ranges));
    }
    if(ranges.size() == 0)
        return BucketMTL(nullptr);
    int64_t numel = this->numel();
    int64_t range_err = fix_ranges(numel, ranges);
    if(range_err != -1){
        utils::throw_exception(false,
            "Error, got invalid splitting range at {$ -> } for tensor with $ elements", 
                    ranges[range_err].first, rnumel);
    
    }
    int64_t total_size = ranges.back().second;
    
    BucketMTL out_bucket(
            BucketMTLStrides{
                this->get_device(),
                make_intrusive<DeviceMTLPrivate>(),
                0,
                total_size
            }, this->dtype_);
    std::get<2>(out_bucket.storage).indexes->allocate_memory(DType::int64, total_size);
    std::vector<int64_t> sizes(ranges.size());

    // turning the ranges into a buffer of long2:
    DeviceMTLShared staging_in;
    staging_in.allocate_memory(sizeof(int64_t) * 2 * ranges.size());
    staging_in.adjust_type_bytes(sizeof(int64_t) * 2);
    std::memcpy(staging_in.get_memory(), &ranges[0], sizeof(int64_t) * 2 * ranges.size());

    
    //loading the shader library
    mtl::abs::MetalContext& ctx = mtl::abs::MetalContext::instance();
    const char** names = {"contiguous_kernel", "affine_kernel", "strided_kernel"};
    std::string kernelName = std::string("ranges_convert_")
                            + std::string(names[this->storage.index()]);

    intrusive_ptr<mtl::abs::Pipeline> pipeline_ = ctx.get_pipeline(kernelName);
    intrusive_ptr<mtl::abs::MetalBuffer> out_indexes = std::get<2>(out_bucket.storage).indexes->get_buffer(); 
    intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = ctx.makeCommandBuffer();
    DeviceMTLPrivate ranges_memory = mtl::mtl_shared_to_private(staging_in, commandBuffer_);
    if(this->is_contiguous()){ // is contiguous
        const auto& s = std::get<0>(this->storage);
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = total_size,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{out_indexes, 0},
                mtl::abs::utils::EncodeBuffer{ranges_memory.get_buffer(), 0},
                static_cast<uint32_t>(ranges.size()),
                total_size,
                total_size,
                s.offset,
                numel
        );
    }
    else if(this->is_affine()){ // is contiguous
        const auto& a = std::get<1>(this->storage);
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = total_size,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{out_indexes, 0},
                mtl::abs::utils::EncodeBuffer{ranges_memory.get_buffer(), 0},
                static_cast<uint32_t>(ranges.size()),
                total_size,
                a.offset,
                static_cast<uint32_t>(a.ndim),
                a.intrusive_sizes,
                a.intrusive_strides,
                numel
        ); 
    }
    else if(this->is_strided()){
        const auto& s = std::get<2>(this->storage);
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = total_size,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{out_indexes, 0},
                mtl::abs::utils::EncodeBuffer{s.indexes->get_buffer(), s.idx_offset * sizeof(int64_t)},
                mtl::abs::utils::EncodeBuffer{ranges_memory.get_buffer(), 0},
                static_cast<uint32_t>(ranges.size()),
                total_size,
                s.nnz
        ); 
    }
    else if(this->is_concatenated()){
        // force this to be strided
        BucketMTL strided = this->bucket_all_indices_mtl();
        const auto& s = std::get<3>(strided.storage);
        mtl::abs::encodeCommand(
                mtl::abs::EncoderOptions{
                    .commandBuffer = commandBuffer_,
                    .pipeline = pipeline_,
                    .size = total_size,
                    .type_bytes = DTypeFuncs::size_of_dtype(this->dtype_)
                },
                mtl::abs::utils::EncodeBuffer{out_indexes, 0},
                mtl::abs::utils::EncodeBuffer{s.buffers[0].indexes->get_buffer(), s.buffers[0].idxOffset},
                mtl::abs::utils::EncodeBuffer{ranges_memory.get_buffer(), 0},
                static_cast<uint32_t>(ranges.size()),
                total_size, nnz
        );

    }
    ctx.run_command(commandBuffer_, /* async = */ true);
    return std::move(out_bucket);  
}

template<>
Tensor BucketMTL::range<Tensor>(std::vector<std::pair<int64_t, int64_t>> ranges) const {
    ArrayVoid out(this->range<BucketMTL>(std::move(ranges)));
    SizeRef ref({out.Size()});
    return Tensor(std::move(out), std::move(ref));
}

void BucketMTL::swap(BucketMTL& bkt){
    std::swap(this->storage, bkt.storage);
    std::swap(this->dtype_, bkt.dtype_);
}

intrusive_ptr<Bucket> BucketMTL::bound_force_contiguity_bucket() const{
    utils::throw_exception(!this->is_null(), "Cannot bound_force_contiguity_bucket null memory");
    return make_intrusive<BucketMTL>(BucketMTLContiguous{this->get_device(), 
                    0, 
                    this->get_device->Size() / DTypeFuncs::size_of_dtype(this->dtype_)
                    }, this->dtype_);
}


intrusive_ptr<Bucket> BucketMTL::force_contiguity_and_bucket() const{
    utils::throw_exception(!this->is_null(), "Cannot force_contiguity_and_bucket null memory");
    return BucketMTL(BucketMTLContiguous{this->get_device(), 
                    0, 
                    this->get_device->Size() / DTypeFuncs::size_of_dtype(this->dtype_)
                    }, this->dtype_)->bucket_all_indices();

}
intrusive_ptr<Bucket> BucketMTL::force_contiguity(int64_t bytes) const{
    utils::throw_exception(!this->is_null(), "Cannot force_contiguity null memory");
    bytes *= DTypeFuncs::size_of_dtype(this->dtype_);
    int64_t total_bytes = this->get_device->Size() / DTypeFuncs::size_of_dtype(this->dtype_);
    utils::throw_exception(bytes <= total_bytes,
            "Error, Expected total bytes to be less than or equal to the available bytes of this device $ ! <= $",
            bytes, total_bytes);
    return make_intrusive<BucketMTL>(BucketMTLContiguous{this->get_device(), 
                    0, 
                    bytes 
                    }, this->dtype_);
}

// is_blocked variable is just for cpu compatibility
intrusive_ptr<Bucket> new_stride_size(int64_t size, bool is_blocked) const{
    auto& s = std::get<2>(this->storage);
    intrusive_ptr<BucketMTL> out_bucket = make_intrusive<BucketMTL>(
            BucketMTLStrides{
            this->get_device(),
            make_intrusive<DeviceMTLPrivate>(),
            0,
            this->numel()
        }, this->dtype_ 
    );
    
    auto& ns = std::get<2>(out_bucket->storage);
    ns.indexes->allocate_memory(s.nnz, DType::Int64); 
    return std::move(out_bucket);
}

intrusive_ptr<Bucket> BucketMTL::copy_strides(bool copy_vals) const {
    if(!copy_vals)
        return this->new_stride_size(this->numel());
    if(!this->is_strided())
        return this->bucket_all_indices_mtl();

    auto& s = std::get<2>(this->storage);
    intrusive_ptr<BucketMTL> out_bucket = make_intrusive<BucketMTL>(
            BucketMTLStrides{
            this->get_device(),
            make_intrusive<DeviceMTLPrivate>(),
            0,
            s.nnz
        }, this->dtype_ 
    );

    
    auto& ns = std::get<2>(out_bucket->storage);
    ns.indexes->allocate_memory(s.nnz, DType::Int64);
    
    if(copy_vals){
        mtl::abs::MetalContext& ctx = mtl::abs::MetalContext::instance();
        intrusive_ptr<mtl::abs::MetalBuffer> out_indexes = std::get<2>(out_bucket->storage).indexes->get_buffer(); 
        intrusive_ptr<mtl::abs::MetalCommand> commandBuffer_ = ctx.makeCommandBuffer();
        MTL::BlitCommandEncoder* blit = commandBuffer_->cmd->blitCommandEncoder();
        blit->copyFromBuffer(s.indexes->get_buffer()->buffer, s.idx_offset, out_indexes->buffer, 0, s.indexes->Size());
        blit->endEncoding();
        ctx.run_command(commandBuffer_, /* async = */ false); 
    }

    return std::move(out_bucket);
}



BucketMTL BucketMTL::from_cpu(const BucketCPU& bkt, MemoryLayout mem_t){
    utils::throw_exception(!bkt.is_null(), "Cannot make a null cpu bucket an mtl bucket");
    BucketCPU c_bkt = bkt.contiguous_cpu();
    BucketMTL out_bucket(c_bkt.numel(), c_bkt.dtype(), mem_t);
    const intrusive_ptr<DeviceHolder>& cpu_devices = c_bkt.intrusive_device();
    const intrusive_ptr<Device>& device = cpu_devices->get(0);
    switch(mem_t){
        case MemoryLayout::Private:{
            // staging into shared
            DeviceMTLShared shared;
            std::ptrdiff_t total_size = c_bkt.storage_size();
            shared.allocate_memory(total_size);
            std::memcpy(shared.get_memory(), device->get_memory(), total_size);
            // finished staging in above
            mtl::abs::MetalContext ctx = mtl::abs::MetalContext::instance();
            intrusive_ptr<mtl::abs::MetalCommand> commandBuffer = ctx.makeCommandBuffer();
            MTL::CommandBuffer* buffer = commandBuffer->cmd;
            MTL::BlitCommandEncoder* blit = buffer->blitCommandEncoder();
            blit->copyFromBuffer(shared.get_buffer()->buffer, 0, out_bucket.get_buffer()->buffer, 0, total_size);
            blit->endEncoding();
            ctx.run_command(commandBuffer, /* async = */ false);
            // finished converting from shared to private above
            return std::move(out_bucket);
        }
        case MemoryLayout::Shared:{
            std::ptrdiff_t total_size = c_bkt.storage_size();
            std::memcpy(out_bucket.data_ptr(), device->get_memory(), total_size);
            return std::move(out_bucket);
        }
        default:
            utils::throw_exception(false,
                    "Cannot convert from cpu with a metal memory layout of $", mem_t);
            return std::move(out_bucket);
    }
    return std::move(out_bucket);
}

}

namespace nt::mtl{

void synchronize(const BucketMTL& bkt){
    if(bkt.is_null()) return;
    if(bkt.is_concatenated()){
        for(const auto& view : std::get<3>(bkt.storage).buffers){
            synchronize(view.buffer);
        }
        return;
    }
    synchronize(bkt.get_buffer());
}

intrusive_ptr<Bucket> synchronize(const intrusive_ptr<BucketMTL>& bkt){
    if(bkt->is_fusing()){
        return bkt->fuse();
    }
    synchronuze(*bkt);
    return intrusive_ptr<Bucket>(bkt);
}

}
