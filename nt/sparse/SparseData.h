#ifndef NT_SPARSE_DATA_H__
#define NT_SPARSE_DATA_H__
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include <iostream>
#include <memory>
#include <cstring>
#include "../utils/api_macro.h"
#include "../memory/meta_allocator.h"

namespace nt{
namespace sparse_details{
class NEUROTENSOR_API SparseMemoryData : public intrusive_ptr_target{
    std::size_t type_size;
    size_t max_size;
    void* memory;
    std::vector<int64_t> indices;
    public:
    SparseMemoryData() = delete;

    SparseMemoryData(std::size_t type_size, int64_t reserve_size = 0)
        : type_size(type_size), max_size(reserve_size), size(0), memory(nullptr) {
        if (reserve_size > 0) {
            memory = MetaMalloc(type_size * reserve_size);
            indices.reserve(reserve_size);
        }
    }

    ~SparseMemoryData() {
        if(memory != nullptr){
            MetaCStyleFree(memory);
            memory = nullptr;
        }
    }
    inline size_t size() const {return indices.size();}
    inline const size_t& capacity() const {return max_size;}

    void reserve(int64_t new_max_size);
    void push_back(int64_t index, const void* data);
    template<typename T>
    inline T& get(int64_t index){
        for (int64_t i = 0; i < size; i++) {
            if (indices[i] == index) {
                return *reinterpret_cast<T*>(static_cast<char*>(memory) + (i * type_size));
            }
        }
        if(size >= max_size){
            reserve(max_size > 0 ? max_size * 2 : 1);
        }
        indices.push_back(index);
        T& out = reinterpret_cast<T*>(memory)[size];
        ++size;
        return out;
    }

    void* access(int64_t index);
    template<typename T>
    inline T* begin_mem() {return reinterpret_cast<T*>(memory);}
    template<typename T>
    inline T* end_mem() {return reinterpret_cast<T*>(memory) + size;}
    template<typename T>
    inline const T* cbegin_mem() const {return reinterpret_cast<const T*>(memory);}
    template<typename T>
    inline const T* cend_mem() const {return reinterpret_cast<const T*>(memory) + size;}

    inline int64_t* begin_indices() {return &indices[0];}
    inline int64_t* end_indices() {return &indices[size];}
    inline const int64_t* cbegin_indices() const {return &indices[0];}
    inline const int64_t* cend_indices() const {return &indices[size];}
    // void sort();
    
};



}

}


#endif
