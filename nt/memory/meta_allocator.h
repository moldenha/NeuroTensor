// There are different device types when it comes to NeuroTensor
// (CPU, CPUShared, [GPUs coming soon], etc)
// Meta is for memory not inside of a tensor
// For example when a shape using ArrayRef is allocated, the default is to use a meta allocator
// The keyword new should not be used inside of NeuroTensor
// [or free, calloc, malloc, etc...]
// The reason this was done is 2 fold:
//  - Memory saftey
//      - A meta allocator automatically maps each ptr to its size
//        But also, marks the total amount of memory allocated each time 
//        This means that during debugging it can be made sure that all memory is freed appropriately
//          TAKE AWAY: [All memory leaks found if used properly]
//  - Standard Allocator
//      - If at any point the actual memory allocator needs to be changed, this is the only
//          file that will need to be changed
//      - Microsoft and other companies have released memory allocators that are faster and better for 
//          memory fragmentation, which could be used for quality optimization and speed ups in the future
//          especially when constructing backward graphs


// Debug mode: [NT_DEBUG_MODE]
//  - Debug mode is used to track exactly where each allocation is made
//    That way if there is a pointer that wasn't deallocated, 
//      during debugging it can be found and the issues can be addressed
//    
//  - Debugging mode has performance issues:
//      - Marking exactly how much memory was allocated and where every time there is a memory allocation
//        creates a lot of overhead, especially to ensure it is thread-safe
//      - For that reason, this marking is only done in debug mode
//      - There is no reason to sacrifice performance unneccessarily once testing has made sure everything works
//      - Otherwise the allocations are just given specific name that can make all alocations easy to change in the future
//  
//  - Note: debug mode should be used with something like the address sanitizer
//
#ifndef NT_META_ALLOCATOR_H__
#define NT_META_ALLOCATOR_H__

#include <atomic> //std::atomic
#include <unordered_map> //std::unordered_map
#include <cstdint> // std::uintptr_t, int64_t
#include <cstdlib> // std::aligned_alloc
#include <cstring> // std::malloc
#include <shared_mutex> // shared_mutex, unique_lock, shared_lock
#include "../utils/always_inline_macro.h" // NT_ALWAYS_INLINE
#include "../utils/numargs_macro.h"

#ifdef _WIN32
#include <malloc.h>  // for _aligned_malloc and _aligned_free
#endif

namespace nt{
namespace utils::memory_details{

class ThreadSafePtrMap{
public:
    using Key = std::uintptr_t;
    using Value = std::tuple<int64_t, std::string, int>;
private:
    std::unordered_map<Key, Value> map;
    mutable std::shared_mutex mtx_; // shared mutex allows multiple readers and one writer
public:
    inline void insert(const Key& key, const Value& value){
        std::unique_lock lock(mtx_);
        map[key] = value;
    }

    inline bool get(const Key& key, Value& value) const {
        std::shared_lock lock(mtx_);
        auto it = map.find(key);
        if(it != map.end()){
            value = it->second;
            return true;
        }
        return false;
    }


    inline void erase(const Key& key){
        std::unique_lock lock(mtx_);
        map.erase(key);
    }

    inline const std::unordered_map<Key, Value>& unsafe_get_cref() const {return map;}
    inline std::unordered_map<Key, Value>& unsafe_get_ref() {return map;}
        
};

inline ThreadSafePtrMap& getMetaRegistry(){
    static ThreadSafePtrMap registry;
    return registry;
}


inline std::atomic<int64_t>& getMetaAllocated(){
    static std::atomic<int64_t> MetaAllocated = 0;
    return MetaAllocated;
}


}

NT_ALWAYS_INLINE void MetaMarkAllocation(void* _ptr, int64_t size, std::string file, int line){
    std::uintptr_t ptr = std::uintptr_t(_ptr);
    utils::memory_details::getMetaAllocated().fetch_add(size, std::memory_order_relaxed);
    utils::memory_details::getMetaRegistry().insert(ptr, std::make_tuple(size, std::move(file), line));
}

NT_ALWAYS_INLINE void MetaMarkDeallocation(void* _ptr){
    std::uintptr_t ptr = std::uintptr_t(_ptr);
    std::tuple<int64_t, std::string, int> _info;
    bool contains = utils::memory_details::getMetaRegistry().get(ptr, _info);
    if(contains){
        int64_t size = std::get<0>(_info);
        utils::memory_details::getMetaAllocated().fetch_sub(size, std::memory_order_relaxed);
        utils::memory_details::getMetaRegistry().erase(ptr);
    }
}

NT_ALWAYS_INLINE int64_t fetch_meta_amt(){return utils::memory_details::getMetaAllocated().load(std::memory_order_acquire);}

// Arrays
#ifdef NT_DEBUG_MODE
template<typename T>
NT_ALWAYS_INLINE T* MetaNewArr_(int64_t size, const char* file, int line){
    T* ptr = new T[size];
    MetaMarkAllocation(ptr, size * sizeof(T), std::string(file), line);
    return ptr;
}

#define MetaNewArr(type, size) MetaNewArr_<type>(size, __FILE__, __LINE__)

template<typename T>
NT_ALWAYS_INLINE void MetaFreeArr(T* arr){
    MetaMarkDeallocation(arr);
    delete[] arr;
}

#else // NT_DEBUG_MODE

#define MetaNewArr(type, size) new type[size]
template<typename T>
NT_ALWAYS_INLINE void MetaFreeArr(T* arr){
    delete[] arr;
}

#endif // NT_DEBUG_MODE 

// Single Pointers
#ifdef NT_DEBUG_MODE
template<typename T, typename... Args>
NT_ALWAYS_INLINE T* MetaNew_(const char* file, int line, Args&&... args){
    T* ptr = new T(std::forward<Args>(args)...);
    MetaMarkAllocation(ptr, sizeof(T), std::string(file), line);
    return ptr;
}

#define MetaNew_Empty_0(type, file, line, ...) MetaNew_<type>(file, line, __VA_ARGS__)
#define MetaNew_Empty_1(type, file, line, ...) MetaNew_<type>(file, line)

#define MetaNew(type, ...) _NT_GLUE_(MetaNew_Empty_, _NT_IS_EMPTY_(__VA_ARGS__))(type, __FILE__, __LINE__, __VA_ARGS__)

template<typename T>
NT_ALWAYS_INLINE void MetaFree(T* ptr){
    MetaMarkDeallocation(ptr);
    delete ptr;
}
#else // NT_DEBUG_MODE

#define MetaNew_Empty_0(type, ...) new type(__VA_ARGS__)
#define MetaNew_Empty_1(type, ...) new type()
#define MetaNew(type, ...) _NT_GLUE_(MetaNew_Empty_, _NT_IS_EMPTY_(__VA_ARGS__))(type, __VA_ARGS__)
template<typename T>
NT_ALWAYS_INLINE void MetaFree(T* ptr){
    delete ptr;
}

#endif // NT_DEBUG_MODE


NT_ALWAYS_INLINE void* untracked_aligned_alloc(std::size_t alignment, int64_t size){
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

NT_ALWAYS_INLINE void untracked_free_aligned_alloc(void* ptr){
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

#ifdef NT_DEBUG_MODE

NT_ALWAYS_INLINE void* MetaAlignedAlloc_(std::size_t alignment, int64_t size, const char* file, int line){
    void* ptr = untracked_aligned_alloc(alignment, size);
    MetaMarkAllocation(ptr, size, std::string(file), line);
    return ptr;
}

#define MetaAlignedAlloc(alignment, size) MetaAlignedAlloc_(alignment, size, __FILE__, __LINE__)

NT_ALWAYS_INLINE void MetaAlignedFree(void* ptr){
    MetaMarkDeallocation(ptr);
    untracked_free_aligned_alloc(ptr);
}


NT_ALWAYS_INLINE void* MetaMalloc_(int64_t size, const char* file, int line){
    void* ptr = std::malloc(size);
    MetaMarkAllocation(ptr, size, std::string(file), line);
    return ptr;
}


#define MetaMalloc(size) MetaMalloc_(size, __FILE__, __LINE__)

NT_ALWAYS_INLINE void MetaCStyleFree(void* ptr){
    MetaMarkDeallocation(ptr);
    std::free(ptr); 
}

#else // NT_DEBUG_MODE
#define MetaAlignedAlloc(alignment, size) untracked_aligned_alloc(alignment, size)
#define MetaAlignedFree(ptr) untracked_free_aligned_alloc(ptr)
#define MetaMalloc(size) std::malloc(size)
#define MetaCStyleFree(ptr) std::free(ptr)
#endif


}


#endif
