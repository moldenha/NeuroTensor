//this is a class to hold a sparse tensor
//the idea is you may have a tensor that holds the majority as zeros for example
//but, you need certain indices to be 1,2,3, etc.
//this is designed for that
//it is fairly limited in functionality currently
#ifndef _NT_SPARSE_TENSOR_H_
#define _NT_SPARSE_TENSOR_H_

namespace nt{
class SparseTensor;
}

#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include <vector>
#include <iostream>

namespace nt{

class SparseMemoryPool : public intrusive_ptr_target {
private:
    struct Block {
        char* data;   // Start of the allocated block
        size_t used;  // Bytes used in this block
        size_t size;  // Total size of the block

        Block(size_t sz) : data(static_cast<char*>(std::malloc(sz))), used(0), size(sz) {
            if (!data) throw std::bad_alloc();
        }

        ~Block() { std::free(data); }
    };

    std::vector<Block*> blocks; // List of memory blocks
    size_t block_size;          // Default block size for expansion

public:
    SparseMemoryPool(size_t init_size = 1024) : block_size(init_size) {
        blocks.push_back(new Block(block_size));
    }

    ~SparseMemoryPool() {
        for (Block* block : blocks) delete block;
    }

    void* allocate(size_t bytes);
    void reset();
};

class SparseTensor{
public:
    using size_value_t = typename SizeRef::ArrayRefInt::value_type;
private:
    Tensor _t;
    intrusive_ptr<SparseMemoryPool> _mem;
    void initialize(Scalar val, SizeRef shape, DType dt, int64_t init_size = 1024);

    inline void collectIntegers(std::vector<int64_t>& a) const {;}
    template<typename... Args>
    inline void collectIntegers(std::vector<int64_t>& a, int64_t i, Args... args) const {
        a.push_back(i);
        collectIntegers(a, args...);
    }

    
    const void* _get(std::vector<size_value_t> xs) const;
    void* _set(std::vector<size_value_t> xs);
    void* _set(const size_value_t& index);
    inline void** stride_begin() const {return _t._vals.stride_begin();}
    inline void** stride_end() const {return _t._vals.stride_end();}
    //important to get memory stored within the tensor
    //usually it just returns stride[0]
    //which in normal cases in fine
    //but because this combines memory from the pool memory, it is important that this memory
    //is directly from that within the tensor that should not be modified
    const void* data_ptr() const { return (*_t._vals.get_bucket().intrusive_device())[0]->get_memory();}

    //this checks to see if the memory in question is in the allocation of memory in the array void
    //otherwise it is in a memory pool
    inline bool in_sparse_memory(const void* ptr) const noexcept {
        return data_ptr() == ptr;
    }


    SparseTensor(const Tensor t, const intrusive_ptr<SparseMemoryPool>& mem);
public:
    SparseTensor(SizeRef sh, DType dt, Scalar val = 0);    
    SparseTensor(Tensor indices, Tensor values, SizeRef sh, DType dt, Scalar val = 0);
    SparseTensor(const SparseTensor&);
    SparseTensor(SparseTensor&&);
    
    SparseTensor& operator=(const SparseTensor&);
    SparseTensor& operator=(SparseTensor&&);


    inline const SizeRef& shape() const {return _t.shape();}
    inline const size_value_t dims() const {return _t.dims();}
    inline const size_value_t& numel() const {return _t.numel();}
    inline const DType& dtype() const {return _t.dtype;}

    template<typename T, typename... Args>
    inline const T& get(int64_t i, Args... args) const {
        std::vector<size_value_t> s;
        collectIntegers(s, i, args...);
        return *reinterpret_cast<const T*>(this->_get(std::move(s)));
    }

    template<typename T, typename... Args>
    inline T& set(int64_t i, Args... args) {
        std::vector<size_value_t> s;
        collectIntegers(s, i, args...);
        return *reinterpret_cast<T*>(this->_set(std::move(s)));
    }
    
    SparseTensor& set(Tensor indices, Tensor values);
    SparseTensor& set(Tensor indices, Scalar value);
    inline Tensor operator>=(const Tensor& i) const {return _t >= i;} 
    inline Tensor operator<=(const Tensor& i) const {return _t <= i;} 
    inline Tensor operator==(const Tensor& i) const {return _t == i;} 
    inline Tensor operator>=(Scalar i) const {return _t >= i;} 
    inline Tensor operator<=(Scalar i) const {return _t <= i;} 
    inline Tensor operator==(Scalar i) const {return _t == i;} 
    inline Tensor operator!=(Scalar i) const {return _t != i;} 
    inline Tensor operator&&(Tensor i) const {return _t && i;} 
    inline Tensor operator||(Tensor i) const {return _t || i;} 
    inline Tensor operator>(const Tensor& i) const {return _t > i;} 
    inline Tensor operator<(const Tensor& i) const {return _t < i;} 
    inline Tensor operator>(Scalar i) const {return _t > i;} 
    inline Tensor operator<(Scalar i) const {return _t < i;} 
    inline SparseTensor operator[](size_value_t i) const {return SparseTensor(_t[i], _mem);}
    inline SparseTensor operator[](const my_range& ra) const {return SparseTensor(_t[ra], _mem);}
    inline SparseTensor operator[](const Tensor& t) const {return SparseTensor(_t[t], _mem);}
    inline SparseTensor operator[](std::vector<my_range> ras) const {return SparseTensor(_t[std::move(ras)], _mem);}
    inline SparseTensor operator[](std::vector<size_value_t> xs) const {return SparseTensor(_t[std::move(xs)], _mem);}
    inline const Tensor& underlying_tensor() const noexcept {return _t;}
    inline SparseTensor transanspose(int64_t a, int64_t b) const {return SparseTensor(_t.transpose(a, b), _mem);}
    inline SparseTensor permute(std::vector<size_value_t> v) const {return SparseTensor(_t.permute(std::move(v)), _mem);}
    inline std::vector<size_value_t> strides() const {return _t.strides();}
    inline SparseTensor flatten(size_value_t a, size_value_t b) const {return SparseTensor(_t.flatten(a, b), _mem);}
    inline SparseTensor unflatten(size_value_t a, size_value_t b) const {return SparseTensor(_t.unflatten(a, b), _mem);}
    inline SparseTensor repeat_(size_value_t amt) const {return SparseTensor(_t.repeat_(amt), _mem);}
    inline SparseTensor repeat_(size_value_t dim, size_value_t amt) const {return SparseTensor(_t.repeat_(amt), _mem);}
    inline Tensor clone() const {return _t.clone();}
    inline Tensor contiguous() const {return _t.clone();}
    SparseTensor& operator ^= (const SparseTensor&);
    SparseTensor& operator ^= (const Tensor&);
};

}

inline std::ostream& operator<<(std::ostream& out, const nt::SparseTensor& t){
    return out << t.underlying_tensor();
}


#endif // _NT_SPARSE_TENSOR_H_
