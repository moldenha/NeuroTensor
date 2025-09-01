#ifndef NT_INTRUSIVE_PTR_INTRUSIVE_TRACKED_LIST_HPP__
#define NT_INTRUSIVE_PTR_INTRUSIVE_TRACKED_LIST_HPP__
#include "intrusive_ptr.hpp"


namespace nt{


namespace track_details{
struct MakeTrackedListAligned {};
template<typename T>
class intrusive_tracked_list_sub : public intrusive_ptr_target {
    T* ptr;
    bool aligned;
public:
    intrusive_tracked_list_sub()
    :ptr(nullptr), aligned(false) {}

    intrusive_tracked_list_sub(int64_t amt)
    :ptr(MetaNewArr(T, amt)), aligned(false) {}

    intrusive_tracked_list_sub(uint64_t amt, const std::size_t align_byte, MakeTrackedListAligned)
    :ptr(nullptr), aligned(true){
        uint64_t size = amt * sizeof(T);
        if (size % align_byte != 0)
            size += align_byte - (size % align_byte);
        /* utils::throw_exception((amt * sizeof(T)) % align_byte == 0, "Cannot
         * align $ bytes", amt * sizeof(T)); */
        ptr = reinterpret_cast<T*>(MetaAlignedAlloc(align_byte, size));
    }

    ~intrusive_tracked_list_sub() {
        if(ptr != nullptr){
            if(aligned){
                MetaAlignedFree(ptr);
            }else{
                MetaFreeArr<T>(this->ptr);
            }
        }
    }

    inline T* get() & noexcept {return ptr;}
    inline const T* get() const & noexcept {return ptr;}

};

}

template<typename T>
class intrusive_tracked_list{
    intrusive_ptr<track_details::intrusive_tracked_list_sub<T>> original;
    T* ptr;
    intrusive_tracked_list(const intrusive_ptr<track_details::intrusive_tracked_list_sub<T>>& _original, T* _ptr)
    :original(_original), ptr(_ptr) {}
public:
    intrusive_tracked_list() noexcept 
    :original(nullptr), ptr(nullptr) {}

    intrusive_tracked_list(std::nullptr_t) noexcept
    :original(nullptr), ptr(nullptr) {}

    explicit intrusive_tracked_list(int64_t amt)
    :original(make_intrusive<track_details::intrusive_tracked_list_sub<T>>(amt)), ptr(nullptr)
    {ptr = original->get();}

    ~intrusive_tracked_list(){
        original.reset();
        ptr = nullptr;
    }

    template <typename U = T,
              typename = std::enable_if_t<!std::is_void<U>::value>>
    inline U &operator*() const noexcept {
        return *(ptr);
    }
    
    inline T* get() const & noexcept {return ptr;}
    // inline const T* get() const & noexcept {return ptr;}

    template <typename IntegerType, typename U = T,
              typename = std::enable_if_t<!std::is_void<U>::value &&
                                          std::is_integral<IntegerType>::value>>
    inline U &operator[](const IntegerType i) const noexcept {
        return ptr[i];
    }

    template <typename IntegerType,
          typename std::enable_if<std::is_integral<IntegerType>::value,
                                  int>::type = 0>
    inline intrusive_tracked_list operator+(const IntegerType i) const noexcept {
        if constexpr (!std::is_void_v<T>) {
            return intrusive_tracked_list(original, ptr + i);
        } else {
            return intrusive_ptr(original, reinterpret_cast<uint8_t *>(ptr) + i);
        }
    }

    inline void nullify() noexcept {
        ptr = nullptr;
        original.nullify();
    }

    inline bool defined() const noexcept {
        return original.defined() && (ptr != nullptr);
    }

    inline operator bool() const noexcept {
        return bool(original) && (ptr != nullptr);
    }
    
    inline void swap(intrusive_tracked_list& other) noexcept {
        std::swap(original, other.original);
        std::swap(ptr, other.ptr);
    }

    inline static intrusive_tracked_list make_aligned(uint64_t amt, const std::size_t align_byte){
        intrusive_ptr<track_details::intrusive_tracked_list_sub<T>> o = 
            make_intrusive<track_details::intrusive_tracked_list_sub<T>>(amt, align_byte, track_details::MakeTrackedListAligned{});
        T* ptr = o->get();
        return intrusive_tracked_list(o, ptr);
    }

};

}

#endif
