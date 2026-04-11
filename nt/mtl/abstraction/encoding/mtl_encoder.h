// this manages encoding all of the function arguments for a metal shaders
#if !defined(NT_MTL_ABSTRACTION_MTL_ENCODER_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_ENCODER_H__

#include "encoding_utils.h"
#include "../mtl_macros.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>


#include "../mtl_buffer.h"
#include "../mtl_command.h"
#include "../mtl_pipeline.h"
#include "../../../intrusive_ptr/intrusive_ptr.hpp"
#include "../../../utils/type_traits.h"
#include "../../../utils/integer_sequence.hpp"
#include "../../../utils/span.hpp"

#include <cstdint>
#include <cstring>




// inline void encodeConcatenatedCommand(EncoderOptions& options,
//                                         EncoderVariable<std::vectorMetalBufferViews>>& buffers_){
//     std::vector<MetalBufferView>& views = buffers_.val;
//     MTL::ComputePipelineReflection* contiguous_reflection = options.contiguous_pipeline->reflection();
//     MTL::ComputePipelineReflection* affine_reflection = options.affine_pipeline->reflection();
//     MTL::CommandBuffer* commandBuffer = options.commandBuffer->cmd;
//     int64_t addingBytes = 0;
//     for(const auto& view : views){
//         bool is_contiguous = (view.strides == nullptr);
//         MTL::Pipeline* pipeline = is_contiguous ? options.contiguous_pipeline->pipeline() : options.affine_pipeline->pipeline();
//         MTL::ComputePipelineReflection* reflection = is_contiguous ? options.contiguous_pipeline->reflection() : options.affine_pipeline->reflection();
//         NS::Array* bindings = reflection->bindings();
//         MTL::ArgumentEncoder* enc_arg = is_contiguous ? options.contiguous_arguments : options.affine_arguments;
//         MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
//         enc_arg->setBuffer(view.buffer->buffer, view.offsetBytes + addingBytes, 0);
//         addingBytes += view.offsetBytes;
//         encoder->setComputePipelineState(pipeline);
//         int64_t index = 0;

//     }
// }




}

// template<typename.... T>
// class EncoderArguments{
//     std::tuple<T...> args;
//     intrusive_ptr<MetalPipeline> pipeline;
    
//     template<std::size_t... integers>
//     inline std::vector<std::reference_wrapper<int64_t>> get_buffer_offsets(utils::index_sequence<integers...>){
//         // this function assumes that they were buffers
//         return std::vector<std::reference_wrapper<int64_t>>{
//             std::ref(std::get<integers>(this->args).offset)...
//         };
//     }
    
//     inline void add_buffer_offset_sub(int64_t adding, EncoderVariable<intrusive_ptr<MetalBuffer>>& buf){
//         buf.offset += (adding * buf.val->typeBytes);        
//     }

//     template<std::size_t... integers>
//     inline void add_buffer_offsets(int64_t adding, utils::index_sequence<integers...>){
//         // this function assumes that they were buffers
//         ((add_buffer_offset_sub(std::get<integers>(this->args), adding)), ...);
//     }

//     // Notice that these setEncoderSub functions the index is not a reference
//     // The reason for that is that the index is already pre-set to the index of T in the tuple, so I don't care if it is incremented
//     inline void setEncoderSub(
//         int64_t index,
//         MTL::ComputeCommandEncoder* encoder,
//         intrusive_ptr<MetalCommand>& cmd,
//         NS::Array* bindings,
//         EncoderVariable<intrusive_ptr<MetalBuffer>>&& arg){
//         details::setEncoderSub(index, encoder, cmd, bindings, std::forward<EncoderVariable<intrusive_ptr<MetalBuffer>>(arg));
//     }
//     inline void setEncoderSub(
//         int64_t index,
//         MTL::ComputeCommandEncoder* encoder,
//         intrusive_ptr<MetalCommand>& cmd,
//         NS::Array* bindings,
//         EncoderVariable<T>&& arg){
//         details::setEncoderSub(index, encoder, cmd, bindings, std::forward<EncoderVariable<T>(arg));
//     }
//     inline void setEncoderSub(
//         int64_t index,
//         MTL::ComputeCommandEncoder* encoder,
//         intrusive_ptr<MetalCommand>& cmd,
//         NS::Array* bindings,
//         EncoderVariable<const T*>&& arg){
//         details::setEncoderSub(index, encoder, cmd, bindings, std::forward<EncoderVariable<T*>(arg));
//     }

//     inline void setEncoderSub(
//         int64_t index,
//         MTL::ComputeCommandEncoder* encoder,
//         intrusive_ptr<MetalCommand>& cmd,
//         NS::Array* bindings,
//         EncoderVariable<intrusive_ptr<MetalBuffer>>&& arg,
//         const MetalBufferView& view){
//         details::setEncoderSub(index, encoder, cmd, bindings, std::forward<EncoderVariable<intrusive_ptr<MetalBuffer>>(arg));
//     }
//     inline void setEncoderSub(
//         int64_t index,
//         MTL::ComputeCommandEncoder* encoder,
//         intrusive_ptr<MetalCommand>& cmd,
//         NS::Array* bindings,
//         EncoderVariable<T>&& arg,
//         const MetalBufferView& view){
//         details::setEncoderSub(index, encoder, cmd, bindings, std::forward<EncoderVariable<T>(arg));
//     }
//     inline void setEncoderSub(
//         int64_t index,
//         MTL::ComputeCommandEncoder* encoder,
//         intrusive_ptr<MetalCommand>& cmd,
//         NS::Array* bindings,
//         EncoderVariable<const T*>&& arg,
//         const MetalBufferView& view){
//         details::setEncoderSub(index, encoder, cmd, bindings, std::forward<EncoderVariable<T*>(arg));
//     }
//     inline void setEncoderSub(
//         int64_t index,
//         MTL::ComputeCommandEncoder* encoder,
//         intrusive_ptr<MetalCommand>& cmd,
//         NS::Array* bindings,
//         EncoderVariable<EncoderViewVar>&& arg,
//         const MetalBufferView& view, const int64_t& total_pn){
//         EncoderViewVar var_type = arg.val;
//         switch(var_type){
//             case EncoderViewVar::Buffer:
//                 details::setEncoderSub(index, encoder, cmd, bindings, 
//                         EncoderVariable<intrusive_ptr<MetalBuffer>>{view.buffer, view.offsetBytes});
//                 return;
//             case EncoderViewVar::Offset:
//                 details::setEncoderSub(index, encoder, cmd, bindings, 
//                         EncoderVariable<int64_t>{view.offsetBytes / view.buffer->typeBytes});
//                 return;
//             case EncoderViewVar::Ndim:
//                 details::setEncoderSub(index, encoder, cmd, bindings, 
//                         EncoderVariable<uint32_t>{static_cast<uint32_t>(view.ndim)});
//                 return;
//             case EncoderViewVar::Numel:
//                 details::setEncoderSub(index, encoder, cmd, bindings, 
//                         EncoderVariable<int64_t>{view.numelBytes / view.buffer->typeBytes});
//                 return;
//             case EncoderViewVar::Sizes:
//                 details::setEncoderSub(index, encoder, cmd, bindings,
//                         EncoderOwning(view.sizes));
//                 return;
//             case EncoderViewVar::Strides:
//                 details::setEncoderSub(index, encoder, cmd, bindings,
//                         EncoderOwning(view.strides));
//                 return;
//             case EncoderViewVar::Indexes:
//                 details::setEncoderSub(index, encoder, cmd, bindings,
//                         EncoderVariable<intrusive_ptr<MetalBuffer>>{view.indexes->get_buffer(), view.idxOffset * sizeof(int64_t)});
//                 return;
//             case EncoderViewVar::TotalPN:
//                 details::setEncoderSub(index, encoder, cmd, bindings,
//                         EncoderVariable<int64_t>{total_pn});
//                 return;
//             default:
//                 return;
//         }
//     }
    
//     template<std::size_t... integers>
//     inline void setEncoder(MTL::ComputeCommandEncoder* encoder, intrusive_ptr<MetalCommand>& cmd,
//                        NS::Array* bindings, utils::integer_sequence<integers...>){
//         static_assert(!type_traits::is_decay_in_v<EncoderVariable<EncoderViewVar>, T...>,
//                 "Error, need to have a metal view to run this encoder");
//         ((setEncoderSub(static_cast<int64_t>(index), encoder, cmd, bindings,
//                 std::get<integers>(args))), ...);
//         ((setEncoderSub(int64_t(integers), encoder, cmd, bindings)), ...);
//     }
    
//     template<std::size_t... integers>
//     inline void setEncoder(MTL::ComputeCommandEncoder* encoder, intrusive_ptr<MetalCommand>& cmd,
//                        NS::Array* bindings, MetalBufferView& view, const int64_t& total_pn, utils::integer_sequence<integers...>){
//         ((setEncoderSub(static_cast<int64_t>(index), encoder, cmd, bindings,
//                 std::get<integers>(args), view, total_pn)), ...);
//     }

//     public:
//         constexpr EncoderArguments() = delete;
//         constexpr EncoderArguments(intrusive_ptr<MetalPipeline> pipeline_, T&&... args_)
//             :pipeline(pipeline_), args(std::make_tuple<T...>(std::forward<T>(args_)...))
//         {}
//         using MetalBufferViewType = typename EncoderVariable<std::vector<MetalBufferView>>;
//         using MetalBufferType = typename EncoderVariable<intrusive_ptr<MetalBuffer>>;
//         using indexes = utils::make_index_sequence<sizeof...(T)>;
//         using buffer_view_indexes = utils::is_same_index_sequence<MetalBufferViewType, T...>;
//         using buffer_indexes = utils::is_same_index_sequence<MetalBufferType, T...>;
//         using all_buffer_indexes = utils::index_sequence_concat<buffer_view_indexes, buffer_indexes>;
//         template<std::size_t I>
//         auto& get() { return std::get<I>(args);}
//         template<std::size_t I>
//         const auto& get() const {return std::get<I>(args);}
//         inline std::vector<std::reference_wrapper<int64_t>> buffer_offsets(){
//             return this->get_buffer_offsets(buffer_indexes{});
//         }
//         inline void add_to_offsets(int64_t num){
//             this->add_buffer_offsets(num, buffer_indexes{});
//         }
//         inline void encodeVariables(MTL::ComputeCommandEncoder* encoder,
//                                 intrusive_ptr<MetalCommand>& cmd,
//                                 NS::Array* bindings, const MetalBufferView& view, const int64_t& total_pn){
//             this->setEncoder(encoder, cmd, bindings, view, total_pn, indexes{});
//         }
//         inline void encodeVariables(MTL::ComputeCommandEncoder* encoder,
//                                 intrusive_ptr<MetalCommand>& cmd,
//                                 NS::Array* bindings){
//             this->setEncoder(encoder, cmd, bindings);
//         }
//         intrusive_ptr<MetalPipeline>& get_pipeline() { return this->pipeline; }
// };


namespace nt::mtl::abs {


// When putting in a concatenated view
// pipeline will be a nullptr, but contiguous and the affine version will be filled in
struct EncoderOptions{
    intrusive_ptr<MetalCommand> commandBuffer;
    intrusive_ptr<MetalPipeline> pipeline;
    int64_t size;
    size_t type_bytes;
};

// EncoderCapture vs EncoderNonOwning:
//  - Note: at one point, I thought that the memory give to the encoder had to be stored somewhere else,
//       setBytes() copies into a temporary internal buffer
//       That buffer is only valid for the encoder’s lifetime
//       If the command buffer is committed later, Metal handles it
//       That is all that is needed, the variables passed in will not be needed for anything else than running the GPU function
//       If they are, it is up to the user to handle it
//

// template<class T>
// struct EncoderVariable{
//     using value_type = T;
//     T val;
// };

// template<class T>
// struct EncoderVariable<T*>{
//     using value_type = T*;
//     T* val;
//     int64_t bytes;
// };

// template<>
// struct EncoderVariable<intrusive_ptr<MetalBuffer>>{
//     using value_type = intrusive_ptr<MetalBuffer>;
//     intrusive_ptr<MetalBuffer> val;
//     int64_t offset;

// };
// template<typename T>
// EncoderVariable<const T*> EncoderOwning(const intrusive_ptr<intrusive_tracked_list_sub<T, false>>>& intrusive_var){
//     return EncoderVariable<const T*>{.val = intrusive_var->get(), .bytes = intrusive_var->get_size() * sizeof(T)};
// }

// // this is to tell the encoder to automatically take the variable from the MetalBufferView
// enum class EncoderViewVar{
//     Buffer,  // where the actual buffer goes
//     Offset,  // offset of the buffer (if it is required)
//     Ndim,    // gives the dimension for affine view (always uint32_t)
//     Numel,   // give the number of elements in the buffer
//     Sizes,   // give the int64_t* sizes for affine view
//     Strides, // give the int64_t* strides for the affine view
//     Indexes, // gibe the int64_t* device buffer for the strided view
//     TotalPN  // This is the total number of elements of the previous buffer(s)
// };





namespace details{





// inline void setEncoder(int64_t& index, 
//                         MTL::ComputerCommandEncoder* encoder, 
//                         intrusuve_ptr<MetalCommand>& cmd,
//                         NS::Array* bindings) {;}

// template<typename T>
// inline void setEncoder(int64_t& index,
//                         MTL::ComputeCommandEncoder* encoder, 
//                         intrusuve_ptr<MetalCommand>& cmd, 
//                         NS::Array* bindings,
//                         T&& arg){
//     setEncoderSub(index, encoder, cmd, bindings, std::forward<T>(arg));
// }

// template<typename T, typename Args>
// inline void setEncoder(int64_t& index, 
//                     MTL::ComputeCommandEncoder* encoder, 
//                     intrusuve_ptr<MetalCommand>& cmd, 
//                     NS::Array* bindings,
//                     T&& arg, Args&&... args){
//     setEncoderSub(index, encoder, cmd, bindings, std::forward<T>(arg));
//     setEncoder(index, encoder, cmd, bindings, std::forward<Args>(args)...);
// }

// is a constant uint3&
inline bool last_arg_grid_encoder(const int64_t& index, 
                                    NS::Array* bindings){
    MTL::Binding* binding = find_index(index, bindings);
    if(!binding) return false;
    MTL::BufferBinding* buf_binding = reinterpret_cast<MTL::BufferBinding*>(binding);
    if(MTL::PointerType* ptype = buf_binding->bufferPointerType()){
        return ptype &&
                binding->type() == MTL::BindingTypeBuffer &&
                buf_binding->bufferDataType() == MTL::DataTypeUInt3 &&
                buf_binding->bufferDataSize() == 12 &&
                ptype->access() == MTL::BindingAccessReadOnly
    }
    return false;
}


template<typename... Args>
using EncoderArguments = typename std::pair<intrusive_ptr<MetalPipeline>, utils::EncoderDispatchSlice<Args...>>;

template<typename... T>
EncoderArguments<T...> makeEncoderArguments(T&&... args){
    return EncoderArguments<T...>{intrusive_ptr<MetalPipeline>(nullptr),
                                    std::forward<T>(args)...};
}

template<typename... T>
EncoderArguments<T...> makeEncoderArguments(intrusive_ptr<MetalPipeline> pipeline, T&&... args){
    return EncoderArguments<T...>{pipeline,
                                    std::forward<T>(args)...};
}

namespace details{

// 0 == contiguous, 1 == affine, 2 == strided
inline size_t memory_view_index(const MetalBufferView& view) noexcept {
    if(view.indexes != nullptr)
        return 2;
    if(view.sizes != nullptr)
        return 1;
    return 0;
}


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


template<typename Slice>
inline auto encodeSliceVariables(Slice& variables, int64_t& size,
                                    intrusive_ptr<MetalCommand>& cmd,
                                    NS::Array* bindings,
                                    MTL::ComputeCommandEncoder* encoder,
                                    const MetalBufferView& view, const int64_t& total_pn){

    auto dispatcher = variables.fix_view_buffer_vars(view, cmd, bindings, total_pn);
    // this is done to encode variables
    dispatcher.apply(subEncodeVar{bindings, encoder});
    size = static_cast<int64_t>(dispatcher.num_variables());
    

}

}

template<typename ContiguousVars, typename AffineVars, typename StridedVars>
inline void encodeCommand(EncoderOptions options, ContiguousVars cvars, AffineVars avars, StridedVars svars, 
                                    ::nt::span<MetalBufferView> views){
    using DIspatchContiguous = typename decltype(cvars.second);
    using DIspatchAffine = typename decltype(avars.second);
    using DIspatchStrided = typename decltype(svars.second);
    utils::throw_exception(options.first != nullptr, "Error: Cannot encode the command with a null pipeline");
    // No reason to check this, there should not be a metal buffer view vector in any of these
    // static_assert(
    //         DispatchContiguous::has_type<std::vector<MetalBufferView>>() == 1 && 
    //         DispatchAffine::has_type<std::vector<MetalBufferView>>() == 1 && 
    //         DispatchStrided::has_type<std::vector<MetalBufferView>>() == 1 && ,
    //         "Error, vars must account for the concatenated view when inputting a concatenated buffer of views"
    // );

    MTL::CommandBuffer* commandBuffer = options.commandBuffer->cmd;
    NS::Array*[3] bindings_ = {cvars.first->reflection()->bindings(),
                              avars.first->reflection()->bindings(),
                              svars.first->reflection()->bindings()
    };
    
    // now going to fix the variables with default fix
    auto cvars_ = cvars.second.fix();
    auto avars_ = avars.second.fix();
    auto svars_ = svars.second.fix();

    
    // Makes sure that all encoders are taken care of and async mode is handled properly
    cvars_.handle_buffer_async(options.commandBuffer, bindings_[0]);
    avars_.handle_buffer_async(options.commandBuffer, bindings_[1]);
    svars_.handle_buffer_async(options.commandBuffer, bindings_[2]);


    int64_t total_pn = 0; // addition of all previous numels
    for(auto& view : views){
        size_t index = details::memory_view_index(view);
        intrusive_ptr<MetalPipeline> pipeline =
                    (index == 0) ? cvars.first
                    : (index == 2) ? svars.first
                    : avars.first;
        MTL::Array* bindings = bindings_[index];
        MTL::ComputeCommandEncoder* encoder = options.commandBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(pipeline->pipeline());


        int64_t size = 0;
        switch(index){
            case 0:
                details::encodeSliceVariables(cvars_, size, options.commandBuffer, bindings, encoder, view, total_pn);
                break;
            case 1:
                details::encodeSliceVariables(avars_, size, options.commandBuffer, bindings, encoder, view, total_pn);
                break;
            case 2:
                details::encodeSliceVariables(svars_, size, options.commandBuffer, bindings, encoder, view, total_pn);
                break;
            default:
                break;
        }
        
        ThreadDispatchConfig config = utils::computeThreadDispatchConfig(view.numelBytes / view.buffer->typeBytes);
        if(details::last_arg_grid_encoder(size, bindings)){
            MTL::Size size_val = config.gridSize;
            encoder->setBytes(&config.gridSize, sizeof(MTL::Size), size);
        }
        encoder->dispatchThreads(config.gridSize, config.threadgroupSize);
        encoder->endEncoding();
        // the offset is automatically adjusted when dealing with different types
        int64_t addingOffset = view.numelBytes / view.buffer->typeBytes;
        total_pn += addingOffset;
        cvars_.add_encoder_buffer_offsets(addingOffset);
        avars_.add_encoder_buffer_offsets(addingOffset);
        svars_.add_encoder_buffer_offsets(addingOffset);
    }
}


template<typename... T>
inline void encodeCommand(EncoderOptions options, T&&... args){
    MTL::ComputePipelineReflection* reflection = options.pipeline->reflection();
    utils::THROW_EXCEPTION(reflection != nullptr, "Error, reflection unable to be handled");
    NS::Array* bindings = reflection->bindings();
    utils::THROW_EXCEPTION(bindings != nullptr, "Error, unable to recieve reflection array");
    constexpr size_t args_size = sizeof...(Args);
    int64_t total_args = bindings->count;
    utils::throw_exception(
            (total_args - 1 == sizeof...(Args))
            || (total_args == sizeof...(Args)),
            "Error, not enough arguments passed for pipeline");
    



    MTL::CommandBuffer* commandBuffer = options.commandBuffer->cmd;
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(options.pipeline->pipeline());

    utils::EncoderDispatchSlice<T...> slice_(std::forward<T>(args)...);
    auto slice = slice.fix();
    slice.handle_buffer_async(options.commandBuffer, bindings);
    
    // encode the variables
    slice.apply(details::subEncodeVars{bindings, encoder});
    std::size_t index = slice.num_variables();

    // encode the size
    ThreadDispatchConfig config = utils::computeThreadDispatchConfig(options.size);
    if(details::last_arg_grid_encoder(static_cast<int64_t>(index), bindings)){
        encoder->setBytes(&config.gridSize, sizeof(MTL::Size), index);
    }
    encoder->dispatchThreads(config.gridSize, config.threadgroupSize);
    encoder->endEncoding();
}


}

#endif
