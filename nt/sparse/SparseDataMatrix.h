#ifndef _NT_SPARSE_DATA_MATRIX_H_
#define _NT_SPARSE_DATA_MATRIX_H_
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include <iostream>
#include <memory>
#include <cstring>
#include <utility>

namespace nt{
namespace sparse_details{
class SparseMemoryMatrixData : public intrusive_ptr_target{
    size_t type_size;
    size_t max_size, size;
    void* memory;
    std::vector<int> col_indices, row_ptrs;
    //row ptrs is basically begin -> (char*)memory + (row_ptrs[row] * type_size) to (char*)memory + (row_ptrs[row+1] * type_size);
    //and translating that into columns is the start of a row's cols is col_indices[row_ptrs[row]] and the last column is col_indices[row_ptrs[row+1]]
    public:
    SparseMemoryMatrixData() = delete;
    SparseMemoryMatrixData(const SparseMemoryMatrixData&) = delete;
    SparseMemoryMatrixData(SparseMemoryMatrixData&& data)
    :type_size(data.type_size), max_size(std::exchange(data.max_size, 0)), size(std::exchange(data.size, 0)), memory(std::exchange(data.memory, nullptr)),
        col_indices(std::move(data.col_indices)), row_ptr(std::move(data.row_ptrs))
    {}
    SparseMemoryMatrixData(size_t type_size, size_t max_size, size_t size, void* mem, std::vector<int> cols, std::vector<int> rows)
    :type_size(type_size), max_size(max_size), size(size), memory(mem), col_indices(cols), row_ptrs(rows)
    {}

    SparseMemoryMatrixData(std::size_t type_size, int64_t reserve_size, std::size_t rows)
        : type_size(type_size), max_size(reserve_size), size(0), memory(nullptr), row_ptrs(rows+1, 0) {
        if (reserve_size > 0) {
            memory = std::malloc(type_size * reserve_size);
            col_indices.reserve(reserve_size);
        }
    }

    ~SparseMemoryMatrixData() {
        if(memory != nullptr){
            std::free(memory);
            memory = nullptr;
        }
    }
    inline size_t size() const {return size;}
    inline const size_t& capacity() const {return max_size;}
    template<typename T = void>
    inline T* data_ptr() {return reinterpret_cast<T*>(memory);}
    template<typename T = void>
    inline const T* data_ptr() const {return reinterpret_cast<T*>(memory);}
    
    inline std::vector<int>& get_cols() {return col_indices;}
    inline const std::vector<int>& get_cols() const {return col_indices;}
    inline std::vector<int>& get_rows() {return row_ptrs;}
    inline const std::vector<int>& get_rows() const {return row_ptrs;}

    template<typename T>
    inline T& get(int row, int col){
        //if the element already exists
        if(row_ptrs[row] != row_ptrs[row+1]){
            auto begin = col_indices.begin() + row_ptrs[row];
            auto end = col_indices.begin() + row_ptrs[row+1];
            int cntr = 0;
            for(; begin != end; ++begin, ++cntr){
                if(*begin == col){
                    reinterpret_cast<T*>(static_cast<char*>(memory) + (row_ptrs[row] * type_size))[cntr];
                }
            }
        }
        //otherwise
        size_t col_cntr = 0;
        if(col_indices.size() == 0){col_indices.push_back(col);} //col_cntr valid at 0
        else{
            auto begin = col_indices.begin() + row_ptrs[row];
            auto end = col_indices.begin() + row_ptrs[row+1];
            //if nothing in that col
            if(begin == end){
                col_indices.insert(begin, col);
            }else{
                for(;begin != end; ++begin, ++col_cntr){
                    if(*begin > col) break;
                }
                //element will be inserted right before the begin element
                col_indices.insert(begin, col);
            }
        }
        //add an element to the row pointers
        std::for_each(row_ptrs.begin() + row+1, row_ptrs.end(), [](int& val){++val;});
        if(size >= max_size){
            size_t new_max_size = max_size > 0 ? max_size * 2 : 1;
            void* new_memory = std::malloc(type_size * new_max_size);
            size_t start = type_size * row_ptrs[row]+col_cntr;
            size_t end = type_size * (size - (row_ptrs[row]+col_cntr));
            void* out_memory = ((char*)new_memory) + start;
            if(start != 0){
                std::memcpy(new_memory, memory, start - type_size);
            }if(end > 0){
                std::memcpy(((char*)new_memory) + start + type_size, ((char*)memory) + start, end);
            }
            ++size;
            std::free(memory);
            memory = new_memory;
            max_size = new_max_size;
            return *reinterpret_cast<T*>(out_memory);
        }
        if(size != 0){
            char* begin = reinterpret_cast<char*>(memory) + ((row_ptrs[row] + col_cntr) * type_size);
            char* end = reinterpret_cast<char*>(memory) + (size * type_size);
            std::copy_backward(begin, end, end + type_size);
        }
        ++size;
        return *reinterpret_cast<T*>(reinterpret_cast<char*>(memory) + ((row_ptrs[row] + col_cntr) * type_size));
    }

    inline void reserve(int64_t new_max_size) {
        if (new_max_size <= max_size) return;
        
        void* new_memory = std::malloc(type_size * new_max_size);
        if (memory) {
            std::memcpy(new_memory, memory, type_size * size);
            std::free(memory);
            memory = new_memory;
            max_size = new_max_size;
        }
    }

    inline SparseMemoryMatrixData&& copy(){
        void* mem = std::malloc(type_size * max_size);
        std::memcpy(mem, memory, type_size * size); 
        return SparseMemoryMatrixData(type_size, max_size, size, mem, col_indices, row_ptrs);
    }


    inline SparseMemoryMatrixData&& block(int row_start, int row_end, int col_start, int col_end){
        std::vector<int> n_row_ptrs(row_end-row_start, 0);
        std::vector<int> n_col_indices;
        n_col_indices.reserve(col_indices.size());
    }
    
};



}

}


#endif
