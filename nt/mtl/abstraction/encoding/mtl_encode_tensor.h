// The rest of the files in the abstractio directory/namespace are meant to work at a lower level included inside of mtl_bucket.cpp
// This specific file is meant to help abstract away an intrusive_ptr<BucketMTL> into a gpu tensor on the kernel
// Currently there is a mtl_encoder header and that is meant for directly handling the memory from the mtl bucket perspective
// This is meant to be a more refined version for interacting with the tensor and making tensor ops easier to use
// This is kind of like how on the cpu the iterator is abstracted away based on the way the memory is stored
// This is going to make the kernel easier to write by just having one gpu tensor struct, and then this is going to make
//  calling that kernel much easier without having to worry about memory layouts except in this one place
// Order:
//      Tensor (any memory layout) -> mtl_encoder_tensor / Bucket (handles/abstracts away layout) -> kernel (any memory layout)
// Think of this as a kind of intermediary to handle memory layout properly and just general tensor indexing

#if !defined(NT_MTL_ABSTRACTION_MTL_ENCODE_TENSOR_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_ENCODE_TENSOR_H__

#include "../mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>


#include "encoding_utils.h"
#include "mtl_encode_bucket.h"
#include "../mtl_arg_encoder.h"
#include "../mtl_buffer.h"
#include "../mtl_command.h"
#include "../mtl_pipeline.h"
#include "../../../intrusive_ptr/intrusive_ptr.hpp"
#include "../../../utils/type_traits.h"
#include "../../../utils/integer_sequence.hpp"
#include "../../../utils/span.hpp"

#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>

namespace nt::mtl::abs {

struct EncoderTensorOptions{
    intrusive_ptr<MetalCommand> commandBuffer;
    intrusive_ptr<MetalPipeline> pipeline;
    int64_t size; // the numel
    bool concat_allowed; // this will change to using a gpu-specific kernel struct for concatted tensors vs just running seperate pipeline/command buffers
};

namespace encoder_tensor_details{

struct EncoderBucketFixer : public ::nt::mtl::abs::utils::BaseEncoderFixer {
    intrusive_ptr<MetalPipeline> pipeline;
    intrusive_ptr<MetalCommand> cmd_buf;
    NS::Array* bindings;
    bool concat_allowed;

   intrusive_ptr<MetalArgEncoder> operator(const intrusive_ptr<BucketMTL>& bkt, std::size_t index) const noexcept {
       if(bkt->is_concatenated){
           if(!concat_allowed) return (*this)(bkt->contiguous(), index);
       }
        auto& ctx = MetalContext::instance();
        intrusive_ptr<MetalArgEncoder> arg_encoder = make_intrusive<MetalArgEncoder>(ctx.device(), pipeline, index);
        EncodeMTLBucket(arg_encoder, bkt, index, bindings cmd_buf, 0);
        cmd_buf->arg_encoders.emplace_back(arg_encoder);
        arg_encoder->finish_encoding();
   }

};

struct subEncodeVar{
    NS::Array* bindings;
    MTL::ComputeCommandEncoder* encoder

    inline MTL::Buffer* check_binding(const std::size_t& index) const {
        MTL::Buffer* binding = utils::find_index(index, bindings);
        utils::throw_exception(binding != nullptr,
                "Error, encoding buffer that does not exist in the kernel");
        utils::THROW_EXCEPTION(
                binding->type() == MTL::BindingTypeBuffer,
                "Error, NeuroTenosr has only been made to use kernels with buffers");
        return binding;
    }
    template<typename T>
    inline void operator()(const ::nt::span<T>& arg, std::size_t index){
        this->check_binding(index);
        this->encoder->setBytes(arg.data(), arg.size_bytes(), index);
    }

    inline void operator()(intrusive_ptr<MetalArgEncoder>& arg, std::size_t index){
        // this was generated from each bucket
        this->check_binding(index);
        this->encoder->setBuffer(arg->get_buffer(), 0, index);
    }

    // Handling explicit buffers:
    inline void operator()(intrusive_ptr<MetalBuffer>& arg, std::size_t index){
        // all of the metal buffers have their async pre-handled
        this->check_binding(index);
        this->encoder->setBuffer(arg->buffer, 0, index);
    }

    inline void operator()(utils::EncodeBuffer arg, std::size_t index){
        // all of the metal buffers have their async pre-handled
        this->check_binding(index);
        this->encoder->setBuffer(arg->buffer, arg.offset, index);
    }

    template<typename T>
    T operator()(const T& var, std::size_t index){
        this->check_binding(index);
        this->encoder->setBytes(&var, sizeof(T), index);
    }
};

inline void extractBuckets_sub(std::vector<intrusive_ptr<BucketMTL>>& bkts){}
template<typename Arg, typename... Args>
inline void extractBuckets_sub(std::vector<intrusive_ptr<BucketMTL>>& bkts, Arg&& arg, Args&&... args){
    if constexpr (type_traits::is_decay_same_v<Arg, intrusive_ptr<BucketMTL>>){
        bkts.push_back(std::forward<Arg>(arg));
        extractBuckets_sub(bkts, std::forward<Args>(args)...);
    }else{
        extractBuckets_sub(bkts, std::forward<Args>(args)...);
    }
}

template<typename... Args>
inline std::vector<intrusive_ptr<BucketMTL>> extractBuckets(Args&&... args){
    std::vector<intrusive_ptr<BucketMTL>> buckets;
    constexpr std::size_t num_buckets = type_traits::count_decay_in_v<intrusive_ptr<BucketMTL>, Args...>;
    if constexpr (num_buckets == 0){
        return buckets;
    }else{
        buckets.reserve(num_buckets);
        extractBuckets_sub(buckets, std::forward<Args>(args)...);
        return std::move(buckets);
    }
}

template<typename Slice>
void dispatchSubSlice(Slice& slice, intrusive_ptr<MetalCommand>& cmd, intrusive_ptr<MetalPipeline>& pipeline, NS::Array* bindings){
    MTL::CommandBuffer* commandBuffer = cmd->cmd;
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline->pipeline());
    
    // previously handled
    // slice.handle_buffer_async(cmd, bindings);
    slice.apply(encoder_tensor_details::subEncodeVars{bindings, encoder}); // encode each variable

    constexpr std::size_t num_arguments = Slice::num_variables();
    ThreadDispatchConfig config = utils::computeThreadDispatchConfig(slice.end() - slice.begin());
    std::size_t index = args_size;
    encoder->setBytes(&config.gridSize, sizeof(MTL::Size), num_arguments);
    
    struct sub_encoding_index_range__{int64_t begin, end};
    sub_encoding_index_range__ d_slice{slice.begin(), slice.end()};
    // since it is just variables, it can be pretty easily encoed by setting the bytes:
    encoder->setBytes(&d_slice, sizeof(sub_encoding_index_range__), index);

    encoder->dispatchThreads(config.gridSize, config.threadgroupSize);
    encoder->endEncoding();

}

}

// for concatenation, the way that it will work is that each one has slices of numel's
// inline std::vector<int64_t> get_numels() const noexcept <- function in BucketMTL to use
// from there, a vector corresponding to (index, begin, end) will be made for each slice 
//
// below is an example when the slices are not contained and the same across all buckets (not completed)
// This is not allowed for now, probably will not be used in the future, but a good thought project
// This wouldn't be allowed because it would create a slow down minimizing the reason for using the GPU
template<typename... Args>
void encodeTensorCommands_concat_general(EncoderTensorOptions options,
                                    std::vector<intrusive_ptr<BucketMTL>>&& bkts,
                                    std::vector<std::size_t>&& concatenated_bkts,
                                    Args&&... args){

    utils::THROW_EXCEPTION(concatenated_bkts.size() != 0, "Error, ecodeTensorCommands_concat should not have been called");
    MTL::ComputePipelineReflection* reflection = options.pipeline->reflection();
    utils::THROW_EXCEPTION(reflection != nullptr, "Error, reflection unable to be handled");
    NS::Array* bindings = reflection->bindings();
    utils::THROW_EXCEPTION(bindings != nullptr, "Error, unable to recieve reflection array");
    constexpr size_t args_size = sizeof...(Args);
    int64_t total_args = bindings->count;
    utils::throw_exception(
            (total_args - 2 == sizeof...(Args)),
            "Error, Kernel is expected to have the number of arguments inputted $, and an additional Slice and uint3 grid argument", sizeof...(Args));     
    
    std::vector<std::vector<int64_t>> numels(concatenated_bkts.size());
    for(size_t i = 0; i < concatented_bkts.size(); ++i) numels[i] = bkts[concatenated_bkts[i]]->get_numels(/*add_prev = */ true);
    int64_t numel = bkts[0]->numel();
    for(const auto& bkt : bkts){
        utils::throw_exception(bkt->numel() == numel, "Error: Expected all buckets for concatenation allowed encoder to have the same number of elements");
    }

    utils::EncoderDispatchSlice<T...> slice_(std::forward<T>(args)...);
    encoder_tensor_details::EncoderBucketFixer fixer{options.pipeline, options.commandBuffer, bindings, options.concat_allowed /*true*/};
    auto slice = slice_.fix(fixer); 
    
    slice.handle_buffer_async(options.commandBuffer, bindings);
    // slice.apply(encoder_tensor_details::subEncodeVars{bindings, encoder});
    
    // for the first slice, it was already that the first tensor in each concatenated group was set as that tensor
    // So now, I am going to find the minimum number of elements, and set that to the first size
    int64_t last_start = 0;
    int64_t last_end = numels[0][0];
    for(const auto& vec : numels) {last_end = std::min(last_end, vec[0]);}
    slice.begin() = last_start;
    slice.end() = last_end;
    dispatchSubSlice(slice, options.commandBuffer, options.pipeline, bindings);

    last_start = last_end;
    int64_t current_end = numel;

    // this is going to be used to create a conversion from the actual indexes to the vector of buckets
    // Since the types are compile-time, this may as well also be compile time so that run times are not slowed further
    using bkt_sequence = utils::is_same_index_sequence<intrusive_ptr<BucketMTL>, Args...>;
    constexpr std::array<std::size_t, bkt_sequence::size> bkt_conv_arr = utils::make_index_array(bkt_sequence{});

    // now to handle the rest
    std::vector<int64_t> current_concat_indexes(concatenated_bkts.size(), 0);
    std::vector<std::pair<std::size_t, int64_t>> update_concats; // the which bucket, and to which concatenated index
    update_concats.reserve(concatenated_bkts.size());
    for(size_t i = 0; i < numels.size(); ++i){
        const int64_t& num = numels[i][current_concat_indexes[i]];
        if(num <= last_end){
            ++current_concat_indexes[i];
            current_end = std::min(current_end, numels[i][current_concat_indexes[i]]);
            update_concats.emplace_back({concatenated_bkts[i], current_concat_indexes[i]});
            continue;
        }
        current_end = std::min(current_end, num);
    }
    // this updates the correct concatenation bucket
    slice.apply(
            [&concatenated_bkts, &update_concats, &bkts, &bkt_conv_arr,
            intrusive_ptr<MetalPipeline>& pipeline = options.pipeline,
            intrusive_ptr<MetalCommand>& cmd_buf = options.commandBuffer](auto& val, std::size_t index){
                if constexpr (type_traits::is_decay_same_v<decltype(val)>, intrusive_ptr<MetalArgEncoder>){
                    const std::size_t original_index = index;
                    std::ssize_t index_ = -1;
                    for(std::size_t i = 0; i < bkt_sequence::size; ++i){
                        if(bkt_conv_arr[i] == index){
                            index_ = i;
                        }
                    }
                    if(index_ == -1) return;
                    index = index_;
                    for(const auto& pair : update_concats){
                        if(index == pair.first){
                            // now update that specific metal encoder
                            auto& ctx = MetalContext::instance();
                            intrusive_ptr<MetalArgEncoder> arg_encoder = make_intrusive<MetalArgEncoder>(ctx.device(), pipeline, original_index);
                            intrusive_ptr<BucketMTL>& bkt = bkts[index];
                            EncodeMTLBucket(arg_encoder, bkt, original_index, bindings cmd_buf, pair.second);
                            cmd_buf->arg_encoders.emplace_back(arg_encoder);
                            arg_encoder->finish_encoding();
                            return;
                        }
                    }
                }
            });
    // then it would be dispatched again, thinking the 
    while(last_end < numel){
        // generalize it for all dispatches
    }



    
}


template<typename... Args>
void encodeTensorCommands_concat(EncoderTensorOptions options,
                                    std::vector<intrusive_ptr<BucketMTL>>&& bkts,
                                    std::vector<std::size_t>&& concatenated_bkts,
                                    Args&&... args){

    utils::THROW_EXCEPTION(concatenated_bkts.size() != 0, "Error, ecodeTensorCommands_concat should not have been called");
    MTL::ComputePipelineReflection* reflection = options.pipeline->reflection();
    utils::THROW_EXCEPTION(reflection != nullptr, "Error, reflection unable to be handled");
    NS::Array* bindings = reflection->bindings();
    utils::THROW_EXCEPTION(bindings != nullptr, "Error, unable to recieve reflection array");
    constexpr size_t args_size = sizeof...(Args);
    int64_t total_args = bindings->count;
    utils::throw_exception(
            (total_args - 2 == sizeof...(Args)),
            "Error, Kernel is expected to have the number of arguments inputted $, and an additional Slice and uint3 grid argument", sizeof...(Args));     
    
    std::vector<int64_t> numels = bkts[concatenated_bkts[0]]->get_numels(true);
    for(size_t i = 1; i < concatenated_bkts.size(); ++i){
        std::vector<int64_t> temp_numels = bkts[concatenated_bkts[i]]->get_numels(true);
        // TODO:
        utils::throw_exception(temp_numels.size() == numels.size(),
                "Error: Currently, concatenation where slice size is different is not allowed, future versions will"
                "handle contiguity for you. For now, when running tensors in functions together that are concatenated"
                "make them contiguous");
        for(size_t j = 0; j < temp_numels.size(); ++j){
           utils::throw_exception(temp_numels[j] == numels[j],
                "Error: Currently, concatenation where slice size is different is not allowed, future versions will"
                "handle contiguity for you. For now, when running tensors in functions together that are concatenated"
                "make them contiguous");
        }
    }
    int64_t numel = bkts[0]->numel();
    for(const auto& bkt : bkts){
        utils::throw_exception(bkt->numel() == numel, "Error: Expected all buckets for concatenation allowed encoder to have the same number of elements");
    }

    utils::EncoderDispatchSlice<T...> slice_(std::forward<T>(args)...);
    encoder_tensor_details::EncoderBucketFixer fixer{options.pipeline, options.commandBuffer, bindings, options.concat_allowed /*true*/};
    auto slice = slice_.fix(fixer); 
    
    slice.handle_buffer_async(options.commandBuffer, bindings);
    // slice.apply(encoder_tensor_details::subEncodeVars{bindings, encoder});
    
    // for the first slice, it was already that the first tensor in each concatenated group was set as that tensor
    // So now, I am going to find the minimum number of elements, and set that to the first size
    slice.begin() = 0;
    slice.end() = numels[0];
    dispatchSubSlice(slice, options.commandBuffer, options.pipeline, bindings);

    // this is going to be used to create a conversion from the actual indexes to the vector of buckets
    // Since the types are compile-time, this may as well also be compile time so that run times are not slowed further
    for(size_t i = 1; i < numels.size(); ++i){
        slice.begin() = numels[i-1];
        slice.end() = numels[i];
        // handle concatenated buckets
        slice.apply_type_function<intrusive_ptr<MetalArgEncoder>, /*GiveIndex*/true, /*GiveTypeIndex*/true>(
        [&bkts, &concatenated_bkts, &i, &bindings,
        intrusive_ptr<MetalPipeline>& pipeline = options.pipeline,
        intrusive_ptr<MetalCommand>& cmd_buf = options.commandBuffer]
        (intrusive_ptr<MetalArgEncoder>& val, std::size_t OriginalIndex, std::size_t TypeIndex){
            // i is the concat index
            // index corresponds to the correct index in the vector of buckets
            for(const auto& concat_idx : concatenated_bkts){
                if(TypeIndex == concat_idx){
                    auto& ctx = MetalContext::instance();
                    intrusive_ptr<MetalArgEncoder> arg_encoder = make_intrusive<MetalArgEncoder>(ctx.device(), pipeline, OriginalIndex);
                    intrusive_ptr<BucketMTL>& bkt = bkts[TypeIndex];
                    EncodeMTLBucket(arg_encoder, bkt, OriginalIndex, bindings cmd_buf, /*concat_index = */i);
                    cmd_buf->arg_encoders.emplace_back(arg_encoder);
                    arg_encoder->finish_encoding();
                    val = arg_encoder;
                }
            }
        });
        dispatchSubSlice(slice, options.commandBuffer, options.pipeline, bindings);

    }
}


template<typename... Args>
void encodeTensorCommands(EncoderTensorOptions options, Args&&... args){
    std::vector<intrusive_ptr<BucketMTL>> bkts = encoder_tensor_details::extractBuckets(std::forward<Args>(args)...);
    std::vector<std::size_t> concatenated_bkts;
    concatenated_bkts.reserve(bkts.size());
    if(!concatenated_bkts.empty() && options.concat_allowed){
        encodeTensorCommands_concat(options, std::move(bkts), std::move(concatenated_bkts), std::forward<Args>(args)...);
        return;
    }

    MTL::ComputePipelineReflection* reflection = options.pipeline->reflection();
    utils::THROW_EXCEPTION(reflection != nullptr, "Error, reflection unable to be handled");
    NS::Array* bindings = reflection->bindings();
    utils::THROW_EXCEPTION(bindings != nullptr, "Error, unable to recieve reflection array");
    constexpr size_t args_size = sizeof...(Args);
    int64_t total_args = bindings->count;
    utils::throw_exception(
            (total_args - 2 == sizeof...(Args)),
            "Error, Kernel is expected to have the number of arguments inputted $, and an additional Slice and uint3 grid argument", sizeof...(Args));    


    MTL::CommandBuffer* commandBuffer = options.commandBuffer->cmd;
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(options.pipeline->pipeline());

    utils::EncoderDispatchSlice<T...> slice_(std::forward<T>(args)...);
    encoder_tensor_details::EncoderBucketFixer fixer{options.pipeline, options.commandBuffer, bindings, options.concat_allowed};
    auto slice = slice_.fix(fixer);
    

    // This will explicitly track the encoding buffers
    // This was already handled during fix for the buckets
    // Honestly, surprised if this is used at all for the bucket dispatchers, but here if needed anyways
    slice.handle_buffer_async(options.commandBuffer, bindings);
    slice.apply(encoder_tensor_details::subEncodeVars{bindings, encoder});
    
    // so now at this point the buffers are encoded
    // it is time to encode the grid, the slice, and the index fixer
    ThreadDispatchConfig config = utils::computeThreadDispatchConfig(options.size);
    std::size_t index = args_size;
    encoder->setBytes(&config.gridSize, sizeof(MTL::Size), index);
    ++index;

    // in this case there is no concatenation, so it is just begining and end of the whole thing
    slice.begin() = 0;
    slice.end() = options.size;

    struct sub_encoding_index_range__{int64_t begin, end};
    sub_encoding_index_range__ d_slice{slice.begin(), slice.end()};

    // since it is just variables, it can be pretty easily encoed by setting the bytes:
    encoder->setBytes(&d_slice, sizeof(sub_encoding_index_range__), index);

    encoder->dispatchThreads(config.gridSize, config.threadgroupSize);
    encoder->endEncoding();
}




}

#endif
