#if !defined(NT_MTL_ABSTRACTION_MTL_ENCODER_UTILS_H__) && defined(NT_MTL_SUPPORTED)
#define NT_MTL_ABSTRACTION_MTL_ENCODER_UTILS_H__

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
#include "../../../refs/ArrayRef.h"
#include "../../../refs/SizeRef.h"

#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <exception>
#include <vector>
#include <atomic>
#include <cstring>
#include <tuple>
#include <utility>

// So each variable needs to be encoded in a specific way
//
// Lists:
//  This includes but is not limited to std::vector and intrusive_tracked_list
//      - they will be turned into an ::nt::span (non owning view that holds the size and memory)

namespace nt::mtl::abs::utils{

// namespace details{

// template<typename T>
// struct is_list_type : type_traits::false_type {};
// // std::vector
// template<typename T, typename A>
// struct is_list_type<std::vector<T, A>> : type_traits::true_type {};
// // nt::span
// template<typename T>
// struct is_list_type<::nt::span<T>> : type_traits::true_type {};
// // intrusive tracked list
// template<typename T>
// struct is_list_type<intrusive_ptr<intrusive_tracked_list_sub<T, false>>> : type_traits::true_type {};
// // ::nt::ArrayRef<T>
// template<typename T>
// struct is_list_type<::nt::ArrayRef<T>> : type_traits::true_type {};
// // ::nt::SizeRef
// template<>
// struct is_list_type<::nt::SizeRef> : type_traits::true_type {};

// template<typename T>
// inline constexpr bool is_list_type_v = is_list_type<T>::value;

// }


MTL::Binding* find_index(const int64_t& index, NS::Array* bindings);
void handle_buffer_sync(intrusive_ptr<MetalCommand>& cmd, intrusive_ptr<MetalBuffer> buf, 
                            NS::Array* bindings, const int64_t& index);

struct EncodeBuffer{
    intrusive_ptr<MetalBuffer> buffer;
    int64_t offset;
};

// there will be a raw encoding for inside of buckets and a tensor based one
// this is to tell the encoder to automatically take the variable from the MetalBufferView
struct ViewBufferArg{
    struct Buffer {
        using type = EncodeBuffer;
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept {
            return EncodeBuffer{buffer.buffer, buffer.offsetBytes};
        }
    };
    struct Offset {
        using type = decltype(MetalViewBuffer::offsetBytes);
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept { return buffer.offsetBytes / buffer.buffer->typeBytes; }
    };
    struct Ndim {
        using type = int32_t; // going to cast to an int32_t
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept {return static_cast<int32_t>(buffer.ndim);}
    };
    struct Numel {
        using type = decltype(MetalBufferView::numelBytes);
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept {return buffer.numelBytes / buffer.buffer->typeBytes;}
    };
    struct Sizes {
        using type = ::nt::span;
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept {return convert_list(buffer.sizes);}
    };
    struct Strides {
        using type = ::nt::span
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept {return convert_list(buffer.strides);
    };
    struct Indexes {
        using type = EncodeBuffer;
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept { 
            return EncodeBuffer{buffer.indexes->get_buffer(), buffer.idxOffset * sizeof(int64_t)};
        }
    };
    struct TotalPN {
        using type = int64_t;
        inline type operator()(const MetalBufferView& buffer, const int64_t& total_pn) noexcept { 
            return total_pn
        }
    };
};

namespace details{
template<typename T>
inline constexpr bool is_view_buffer_var_v = type_traits::is_in_v<
                T, ViewBufferArg::Buffer,  
                    ViewBufferArg::Offset, ViewBufferArg::Ndim, ViewBufferArg::Numel, ViewBufferArg::Sizes,
                    ViewBufferArg::Strides, ViewBufferArg::Indexes, ViewBufferArg::TotalPN>; 

template<typename T, std::size_t N>
constexpr std::size_t constexpr_find_index(
    const std::array<T, N>& arr,
    const T& value)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (arr[i] == value)
            return i;
    }
    return N; // sentinel for "not found"
}

}

// This is meant to act as a Base class for encoding types that need to be automatically converted
// If you have multiple types to fix, you can just run a single function that has a struct which inherits from this one for example:
/*
 
// in this example I am only fixing intrusive_ptr<BucketMTL>:
struct BucketFixer : public utils::BaseEncodeFixer {
    intrusive_ptr<MetalBuffer> operator()(intrusive_ptr<BucketMTL>& bkt, std::size_t index) const noexcept { return ...}
};

 */
struct BaseEncodeFixer{
    template<typename T>
    T operator()(const T& v, std::size_t index) const noexcept { return v;}
    
    template<typename T>
    inline ::nt::span<T> operator()(const std::vector<T>& vec, std::size_t) const noexcept { return ::nt::span<T>(vec); }
    
    template<typename T>
    inline ::nt::span<T> operator()(const ::nt::span<T>& s, std::size_t) const noexcept {return s;}

    template<typename T>
    inline ::nt::span<T> operator()(const intrusive_ptr<intrusive_tracked_list_sub<T, false>>& l, std::size_t) const noexcept { 
        return ::nt::span<T>(l->get(), l->get_size()); 
    }
    
    template<typename T>
    inline ::nt::span<T> operator()(const ::nt::ArrayRef<T>& l, std::size_t) const noexcept { 
        return ::nt::span<T>(l.cbegin(), l.size()); 
    }

    inline ::nt::span<::nt::SizeRef::value_type> operator()(const ::nt::SizeRef& l, std::size_t) const noexcept { 
        return ::nt::span<T>(l.cbegin(), l.size()); 
    }


};

template<typename... Args>
class EncoderDispatchSlice{
    std::tuple<Args...> tup;
    int64_t begin_, end_;
    
    template<std::size_t Index>
    inline auto remake_viewBufferVar(const MetalBufferView& view, intrusive_ptr<MetalCommand>& cmd
                                    NS::Array* binding, const int64_t& total_pn) noexcept {
        using type = std::tuple_element_t<Index, std::tuple<Args...>>;
        if constexpr (details::is_view_buffer_var_v<type>){
            type t;
            if constexpr (type_traits::is_same_v<type_traits::remove_cvref_t<type>, ViewBufferArg::Buffer){
                // synchronize the buffer
                handle_buffer_sync(cmd. view.buffer, binding, Index);
            }
            return t(view);
        }else{
            return std::get<Index>(tup);
        }
    }

    template<std::size_t... Integers>
    inline auto remake_tuple_viewBufferVar(const MetalBufferView& view, intrusive_ptr<MetalCommand>& cmd,
                                            NS::Array* Binding, const int64_t& total_pn, utils::index_sequence<Integers...>) noexcept {
        return std::make_tuple(remake_viewVar<Integers>(view, cmd, Binding, total_pn) ...);
    }

    template<typename Type, bool GiveIndex, bool GiveTypeIndex, typename Func, std::size_t Index>
    inline void applyTypeFunctionSub(Func&& f) noexcept {
        using type = std::tuple_element_t<Index, std::tuple<Args...>>;
        if constexpr (type_traits::is_decay_same_v<type, Type>){
            if constexpr (GiveIndex){
                if constexpr (GiveTypeIndex){
                    // this gives the index relative to all the same types (for example the 2nd intrusive_ptr<MetalBuffer>)
                    using cur_sequence = utils::is_same_index_sequence<Type, Args...>;
                    constexpr std::array<std::size_t, cur_sequence::size> cur_sequence_arr = utils::make_index_array(cur_sequence{});
                    constexpr std::size_t TypeIndex = constexpr_find_index(cur_sequence_arr, Index);
                    static_assert(TypeIndex != cur_sequence::size, "Internal logic error: type index not calculated correctly");
                    f(std::get<Index>(tup), Index, TypeIndex);
                }else{
                    f(std::get<Index>(tup), Index);
                }
            }else{
                f(std::get<Index>(tup));
            }
        }
    }

    template<typename Type, bool GiveIndex, bool GiveTypeIndex, typename Func, std::size_t... Indexes>
    inline void applyTypeFunction(Func&& f, utils::index_sequence<Indexes...>) noexcept {
        (applyTypeFunctionSub<Type, GiveIndex, GiveTypeIndex, Func&, Indexes>(f), ...);
    }
    
    // auto is used because it is pretty obviously going to be the OutType or the input type
    template<typename InType, typename OutType, bool GiveIndex, typename Func, std::size_t Index>
    inline auto convertTypeFunctionSub(Func&& f) const noexcept {
        using type = std::tuple_element_t<Index, std::tuple<Args...>>;
        if constexpr (type_traits::is_decay_same_v<type, InType>){
            if constexpr (GiveIndex){
                return f(std::get<Index>(tup), Index);
            }else{
                return f(std::get<Index>(tup));
            }
        }else{
            return std::get<Index>(tup);
        }
    }

    template<typename InType, typename OutType, bool GiveIndex, typename Func, std::size_t... Indexes>
    inline auto convertTypeFunction(Func&& f, utils::index_sequence<Indexes...>) const noexcept {
        if constexpr (GiveIndex){
            static_assert(type_traits::is_decay_same_v<OutType, std::result_of<Func(InType, std::size_t)>::type,
                    "Error, expected a specific out type for converting a function type");
        }else{
            static_assert(type_traits::is_decay_same_v<OutType, std::result_of<Func(InType)>::type,
                    "Error, expected a specific out type for converting a function type");
        }
        return std::make_tuple(convertTypeFunctionSub<Type, GiveIndex, Func&, Indexes>(f) ...);
    }

    template<class T, std::size_t... Indexes>
    inline auto convertStructFunction(T&& s, utils::index_sequence<Indexes...>) const noexcept {
        return std::make_tuple(std::forward<T>(s)(std::get<Indexes>(tup), Indexes) ...);
    }

    template<class T, std::size_t... Indexes>
    inline void applyStructFunction(T&& s, utils::index_sequence<Indexes...>) noexcept {
        (std::forward<T>(s)(std::get<Indexes>(tup), Indexes), ...);
        
    }

    public:
        using tuple_type = std::tuple<Args...>;
        using index_seq = utils::make_index_sequence<sizeof...(Args)>;
        EncoderDispatchSlice() = delete;
        EncoderDispatchSlice(std::tuple<Args...> tup_, int64_t begin = 0, int64_t end = -1) noexcept
            :tup(std::move(tup_)), begin_(begin), end_(end)
        {}
        EncoderDispatchSlice(Args&&... args) noexcept
            :tup(std::make_tuple(std::forward<Args>(args)...)), begin_(0), end_(-1)
        {}

        inline auto fix_view_buffer_vars(const MetalBufferView& view, intrusive_ptr<MetalCommand>& cmd, 
                                                NS::Array* bindings, const int64_t& total_pn) noexcept {
            return EncoderDispatchSlice<Args...>::from_tuple(
                    this->remake_tuple_viewBufferVar(view, cmd, bindings, total_pn, index_seq {}),
                    this->begin_, this->end_
            );
        }

        template<typename T, bool GiveIndex, bool GiveTypeIndex, typename Func>
        inline void apply_type_function(Func&& f){
            this->applyTypeFunction<T, GiveIndex, GiveTypeIndex>(std::forward<Func>(f), index_seq {});
        }

        template<typename InType, typename OutType, bool GiveIndex, typename Func>
        inline auto convert_type_function(Func&& f) const noexcept {
            return EncoderDispatchSlice<Args...>::from_tuple( 
                this->convertTypeFunction<InType, OutType, GiveIndex>(std::forward<Func>(f), 
                    utils::make_index_sequence<sizeof...(Args)>{}),
                this->begin_, this->end_
            );
        }

        inline void handle_buffer_async(intrusive_ptr<MetalCommand>& cmd, NS::Array* bindings) noexcept {
            this->apply_type_function<intrusive_ptr<MetalBuffer>, true, false>(
                    [&cmd, &bindings](intrusive_ptr<MetalBuffers> buf, int64_t index){
                        handle_buffer_sync(cmd, buf, bindings, index);
                    }
            );
            this->apply_type_function<EncodeBuffer, true, false>(
                    [&cmd, &bindings](EncodeBuffer& buf, int64_t index){
                        handle_buffer_sync(cmd, buf.buffer, bindings, index);
                    }
            );
        }

        inline void add_encoder_buffer_offsets(const int64_t& val) noexcept {
            this->apply_type_function<EncodeBuffer, false, false>(
                    [&val](EncodeBuffer& buf){
                        buf.offset += (val * buf.buffer->typeBytes);
                    }
            );
        }

        template<class T>
        inline auto fix(T&& s) const noexcept {
            return EncoderDispatchSlice<Args...>::from_tuple(
                this->convertStructFunction(std::forward<T>(s), index_seq {}),
                this->begin_, this->end_
            );

        }
        inline auto fix() const noexcept {
            BaseEncodeFixer fixer;
            return this->fix(fixer);
        }

        template<typename T>
        inline void apply(T&& s) noexcept {
            this->applyStructFunction(std::forward<T>(s), index_seq{});
        }
        inline int64_t& begin() noexcept {return this->begin_;}
        inline int64_t& end() noexcept {return this->end_;}
        inline const int64_t& begin() const noexcept {return this->begin_;}
        inline const int64_t& end() const noexcept {return this->end_;}
        
        template<typename... ArgsT>
        static inline EncoderDispatchSlice<ArgsT...> from_tuple(std::tuple<ArgsT...> tup, int64_t begin = 0, int64_t end = -1) noexcept {
            return EncoderDispatchSlice(std::move(tup), begin, end);
        }

        template<typename T>
        static inline constexpr std::size_t has_type() const noexcept {return utils::is_same_index_sequence<T, Args...>::size;}
        static constexpr std::size_t num_variables() const noexcept {std::tuple_size_v<tuple_type>;}
};

}

#endif
