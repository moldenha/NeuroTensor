#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../dtype/ArrayVoid.hpp"
#include "SparseTensor.h"
#include "../utils/utils.h"
#include "../utils/macros.h"
#include <vector>
#include <cstdlib>

namespace nt{

void* SparseMemoryPool::allocate(size_t bytes) {
    if (bytes > block_size / 2) { 
        // Large allocation: use a separate block
        Block* large_block = new Block(bytes);
        blocks.push_back(large_block);
        return large_block->data;
    }

    // Find a block with enough space
    for (Block* block : blocks) {
        if (block->used + bytes <= block->size) {
            void* ptr = block->data + block->used;
            block->used += bytes;
            return ptr;
        }
    }

    // No space, create a new block
    Block* new_block = new Block(block_size);
    blocks.push_back(new_block);
    void* ptr = new_block->data;
    new_block->used = bytes;
    return ptr;
}

void SparseMemoryPool::reset() {
    // Reset all blocks for reuse
    for (Block* block : blocks) block->used = 0;
}

void SparseTensor::initialize(Scalar val, SizeRef shape, DType dt, int64_t init_size){
    utils::throw_exception(dt != DType::TensorObj, "DType TensorObj is not supported for sparse tensors");
    ArrayVoid out_a(1, dt);
    out_a = val;
    ArrayVoid out = out_a.new_strides(shape.multiply());
    void* ptr = (out.get_bucket().intrusive_device())[0]->get_memory(); //setting it to the correct memory
    void** begin = out.stride_begin();
    void** end = out.stride_end();
    for(;begin != end; ++begin)
        *begin = ptr;
    _t = Tensor(std::move(out), std::move(shape));
    _mem = make_intrusive<SparseMemoryPool>(init_size);
}


const void* SparseTensor::_get(std::vector<size_value_t> xs) const {
    utils::THROW_EXCEPTION(
            xs.size() == dims(),
            "When getting indices from a sparse tensor, must get a single scalar",
            dims(), xs.size());

    size_value_t xs_size = static_cast<size_value_t>(xs.size());
    for (size_value_t i = 0; i < xs_size; ++i) {
        xs[i] = xs[i] < 0 ? xs[i] + dims() : xs[i];
    }

    int64_t index = 0;
    int64_t stride = 1;
    auto begin = xs.begin();
    auto end = xs.end();
    SizeRef n_size = shape();
    for (int64_t i = xs.size() - 1; i >= 0; --i) {
        index += xs[i] * stride;
        stride *= n_size[i];
    }
    utils::throw_exception(index < numel(), "Expected to have index less than $ when flattened, but got index of $ out of bounds error", numel(), index);
    void** val_array = _t._vals.stride_begin();
    const void* ptr = val_array[index];
    return ptr;
}

void* SparseTensor::_set(std::vector<size_value_t> xs) {
    utils::THROW_EXCEPTION(
            xs.size() == dims(),
            "When getting indices from a sparse tensor, must get a single scalar",
            dims(), xs.size());

    size_value_t xs_size = static_cast<size_value_t>(xs.size());
    for (size_value_t i = 0; i < xs_size; ++i) {
        xs[i] = xs[i] < 0 ? xs[i] + dims() : xs[i];
    }

    int64_t index = 0;
    int64_t stride = 1;
    auto begin = xs.begin();
    auto end = xs.end();
    SizeRef n_size = shape();
    for (int64_t i = xs.size() - 1; i >= 0; --i) {
        index += xs[i] * stride;
        stride *= n_size[i];
    }
    utils::throw_exception(index < numel(), "Expected to have index less than $ when flattened, but got index of $ out of bounds error", numel(), index);
    void** val_array = _t._vals.stride_begin();
    void* ptr = val_array[index];
    if(in_sparse_memory(ptr)){
        val_array[index] = _mem->allocate(DTypeFuncs::size_of_dtype(_t.dtype()));
        ptr = val_array[index];
    }
    return ptr;
}

void* SparseTensor::_set(const size_value_t& index) {
    utils::THROW_EXCEPTION(
            index < numel(),
            "Expected to set with index less than numel but got $ for numel $",
            index, numel());

    void** begin = _t._vals.stride_begin();
    void* ptr = begin[index];
    if(in_sparse_memory(ptr)){
        begin[index] = _mem->allocate(DTypeFuncs::size_of_dtype(_t.dtype()));
        ptr = begin[index];
    }
    return ptr;
}


SparseTensor::SparseTensor(const Tensor t, const intrusive_ptr<SparseMemoryPool>& mem)
    :_t(t), _mem(mem)
{}

SparseTensor::SparseTensor(SizeRef sh, DType dt, Scalar val)
    :_t(nullptr), _mem(nullptr)
{
    this->initialize(val, sh, dt);
}


//this needs to be updated to work like t[indices] = values;
SparseTensor::SparseTensor(Tensor indices, Tensor values, SizeRef sh, DType dt, Scalar val)
    :_t(nullptr), _mem(nullptr)
{
    this->initialize(val, sh, dt);
    void* out_ptr_data = this->_mem->allocate((values.numel()+1) * DTypeFuncs::size_of_dtype(dt));
    utils::THROW_EXCEPTION(values.dtype() == _t.dtype(), "Expected to get same dtype for values ($) as the sparse tensor was initialized with ($)", values.dtype(), _t.dtype());
    utils::THROW_EXCEPTION(
        indices.dtype() == DType::TensorObj,
        "RuntimeError: expected DType TensorObj for indices when setting sparse tensor but got $", indices.dtype());
    utils::THROW_EXCEPTION(
        indices.is_contiguous(),
        "RuntimeError: Expected indexing tensor to be contiguous");
    utils::THROW_EXCEPTION(
        indices.numel() == dims(),
        "Expected indexing tensor to have $ tensors but had $", dims(),
        indices.numel());
    const Tensor *begin = reinterpret_cast<const Tensor *>(indices.data_ptr());
    const Tensor *end = begin + indices.numel();
    const Tensor *begin_cpy = begin;
    int64_t num_elements = begin->numel();
    utils::THROW_EXCEPTION(num_elements == values.numel(), "Expected to get same number of elements for values ($) as indices ($)",values.numel(), num_elements);
    for (; begin != end; ++begin){
        utils::THROW_EXCEPTION(
            begin->dtype() == DType::int64 && begin->is_contiguous(),
            "Expected indexing tensor to have dtype int64 but got $ and expected to be contiguous",
            begin->dtype());
        utils::THROW_EXCEPTION(num_elements = begin->numel(), 
                               "Expected all coordinate tensors in indices to have same size but got $ and $", 
                               num_elements, begin->numel());
        
    }
    begin = begin_cpy;

    // making the strides for indexing:
    const std::vector<size_value_t> s = strides();
    std::vector<size_value_t> ns(s.size());
    std::copy(s.begin(), s.end(), ns.begin());

    // keeping track of each int64_t pointer for the indexing
    NT_VLA(const size_value_t*, ptrs, dims());
    // const size_value_t *ptrs[dims()];
    size_value_t i = 0;
    for (; begin != end; ++begin, ++i) {
        ptrs[i] = reinterpret_cast<const int64_t *>(begin->data_ptr());
    }
    // making a new ArrayVoid to keep track of all the indices
    const size_value_t &n_size = begin_cpy->numel();
    void** my_begin = stride_begin();
    void** my_end = stride_end();
    const size_value_t &num_dims = indices.numel();

    values._vals.cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >([&out_ptr_data, &my_begin, &my_end, &n_size, &ptrs, &ns, &num_dims](auto val_begin, auto val_end){
        using value_t = utils::IteratorBaseType_t<decltype(val_begin)>;
        value_t* my_out_ptr_data = reinterpret_cast<value_t*>(out_ptr_data);
        // finding each index
        for (size_value_t i = 0; i < n_size; ++i, ++my_out_ptr_data, ++val_begin) {
            // getting the ith index to copy
            size_value_t index = 0;
            for (size_value_t j = 0; j < num_dims - 1; ++j) {
                index += ptrs[j][i] * ns[j + 1];
            }
            index += ptrs[num_dims - 1][i];
            *my_out_ptr_data = *val_begin;
            my_begin[index] = my_out_ptr_data;
        }

    });
    NT_VLA_DEALC(ptrs);
    
}

SparseTensor::SparseTensor(const SparseTensor& t)
    :_t(t._t), _mem(t._mem)
{}


SparseTensor::SparseTensor(SparseTensor&& t)
    :_t(std::move(t._t)), _mem(std::move(t._mem))
{}

SparseTensor& SparseTensor::operator=(const SparseTensor& t){
    _t.nullify();
    _t = t._t;
    _mem = t._mem;
    return *this;
}

SparseTensor& SparseTensor::operator=(SparseTensor&& t){
    _t = std::move(t._t);
    _mem = std::move(t._mem);
    return *this;
}


//this needs to be updated to work like t[indices] = values;
SparseTensor& SparseTensor::set(Tensor indices, Tensor values){    
    void* out_ptr_data = this->_mem->allocate((values.numel()+1) * DTypeFuncs::size_of_dtype(_t.dtype()));
    utils::THROW_EXCEPTION(values.dtype() == _t.dtype(), 
                           "Expected to get same dtype for values ($) as the sparse tensor was initialized with ($)", 
                           values.dtype(), _t.dtype());
    utils::THROW_EXCEPTION(
        indices.dtype() == DType::TensorObj,
        "RuntimeError: expected DType TensorObj for indices when setting sparse tensor but got $", indices.dtype());
    utils::THROW_EXCEPTION(
        indices.is_contiguous(),
        "RuntimeError: Expected indexing tensor to be contiguous");
    utils::THROW_EXCEPTION(
        indices.numel() == dims(),
        "Expected indexing tensor to have $ tensors but had $", dims(),
        indices.numel());
    const Tensor *begin = reinterpret_cast<const Tensor *>(indices.data_ptr());
    const Tensor *end = begin + indices.numel();
    const Tensor *begin_cpy = begin;
    int64_t num_elements = begin->numel();
    utils::THROW_EXCEPTION(num_elements == values.numel(), "Expected to get same number of elements for values ($) as indices ($)",values.numel(), num_elements);
    for (; begin != end; ++begin){
        utils::THROW_EXCEPTION(
            begin->dtype() == DType::int64 && begin->is_contiguous(),
            "Expected indexing tensor to have dtype int64 but got $ and expected to be contiguous",
            begin->dtype());
        utils::THROW_EXCEPTION(num_elements = begin->numel(), 
                               "Expected all coordinate tensors in indices to have same size but got $ and $", 
                               num_elements, begin->numel());
        
    }
    begin = begin_cpy;

    // making the strides for indexing:
    const std::vector<size_value_t> s = strides();
    std::vector<size_value_t> ns(s.size());
    std::copy(s.begin(), s.end(), ns.begin());

    // keeping track of each int64_t pointer for the indexing
    NT_VLA(const size_value_t*, ptrs, dims());
    // const size_value_t *ptrs[dims()];
    size_value_t i = 0;
    for (; begin != end; ++begin, ++i) {
        ptrs[i] = reinterpret_cast<const int64_t *>(begin->data_ptr());
    }
    // making a new ArrayVoid to keep track of all the indices
    const size_value_t &n_size = begin_cpy->numel();
    void** my_begin = stride_begin();
    void** my_end = stride_end();
    const size_value_t &num_dims = indices.numel();

    values._vals.cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >([&out_ptr_data, &my_begin, &my_end, &n_size, &ptrs, &ns, &num_dims](auto val_begin, auto val_end){
        using value_t = utils::IteratorBaseType_t<decltype(val_begin)>;
        value_t* my_out_ptr_data = reinterpret_cast<value_t*>(out_ptr_data);
        // finding each index
        for (size_value_t i = 0; i < n_size; ++i, ++my_out_ptr_data, ++val_begin) {
            // getting the ith index to copy
            size_value_t index = 0;
            for (size_value_t j = 0; j < num_dims - 1; ++j) {
                index += ptrs[j][i] * ns[j + 1];
            }
            index += ptrs[num_dims - 1][i];
            *my_out_ptr_data = *val_begin;
            my_begin[index] = my_out_ptr_data;
        }

    });
    NT_VLA_DEALC(ptrs);
    return *this;
}
//this needs to be updated to work like t[indices] = values;
SparseTensor& SparseTensor::set(Tensor indices, Scalar value){

    utils::THROW_EXCEPTION(
        indices.dtype() == DType::TensorObj,
        "RuntimeError: expected DType TensorObj for indices when setting sparse tensor but got $", indices.dtype());
    utils::THROW_EXCEPTION(
        indices.is_contiguous(),
        "RuntimeError: Expected indexing tensor to be contiguous");
    utils::THROW_EXCEPTION(
        indices.numel() == dims(),
        "Expected indexing tensor to have $ tensors but had $", dims(),
        indices.numel());
    const Tensor *begin = reinterpret_cast<const Tensor *>(indices.data_ptr());
    const Tensor *end = begin + indices.numel();
    const Tensor *begin_cpy = begin;
    int64_t num_elements = begin->numel();
    void* out_ptr_data = this->_mem->allocate((num_elements+1) * DTypeFuncs::size_of_dtype(_t.dtype()));
    for (; begin != end; ++begin){
        utils::THROW_EXCEPTION(
            begin->dtype() == DType::int64 && begin->is_contiguous(),
            "Expected indexing tensor to have dtype int64 but got $ and expected to be contiguous",
            begin->dtype());
        utils::THROW_EXCEPTION(num_elements = begin->numel(), 
                               "Expected all coordinate tensors in indices to have same size but got $ and $", 
                               num_elements, begin->numel());
        
    }
    begin = begin_cpy;

    // making the strides for indexing:
    const std::vector<size_value_t> s = strides();
    std::vector<size_value_t> ns(s.size());
    std::copy(s.begin(), s.end(), ns.begin());

    // keeping track of each int64_t pointer for the indexing
    NT_VLA(const size_value_t*, ptrs, dims());
    // const size_value_t *ptrs[dims()];
    size_value_t i = 0;
    for (; begin != end; ++begin, ++i) {
        ptrs[i] = reinterpret_cast<const int64_t *>(begin->data_ptr());
    }
    // making a new ArrayVoid to keep track of all the indices
    const size_value_t &n_size = begin_cpy->numel();
    void** my_begin = stride_begin();
    void** my_end = stride_end();
    const size_value_t &num_dims = indices.numel();

    _t._vals.cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >([&out_ptr_data, &my_begin, &my_end, &n_size, &ptrs, &ns, &num_dims, &value](auto val_begin, auto val_end){
        using value_t = utils::IteratorBaseType_t<decltype(val_begin)>;
        value_t* my_out_ptr_data = reinterpret_cast<value_t*>(out_ptr_data);
        value_t s_val = value.to<value_t>();
        // finding each index
        for (size_value_t i = 0; i < n_size; ++i, ++my_out_ptr_data, ++val_begin) {
            // getting the ith index to copy
            size_value_t index = 0;
            for (size_value_t j = 0; j < num_dims - 1; ++j) {
                index += ptrs[j][i] * ns[j + 1];
            }
            index += ptrs[num_dims - 1][i];
            *my_out_ptr_data = s_val;
            my_begin[index] = my_out_ptr_data;
        }

    });
    NT_VLA_DEALC(ptrs);
    return *this;
}

inline Scalar ptr_to_scalar(const Tensor& t, const void* val){
    return t.arr_void().cexecute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool> > >([&val](auto begin, auto end){
        using value_t = utils::IteratorBaseType_t<decltype(begin)>;
        value_t co_val = *reinterpret_cast<const value_t*>(val);
        return Scalar(co_val);
    });
}


SparseTensor& SparseTensor::operator ^= (const SparseTensor& val){
    utils::THROW_EXCEPTION(val.dtype() == dtype(), "Expected dtypes of input ($) and this tensor ($) to match",
                           val.dtype(), dtype());
    utils::THROW_EXCEPTION(val.shape() == shape(), "Expected shapes of input ($) and this tensor ($) to match",
                           val.shape(), shape());
    utils::THROW_EXCEPTION(DTypeFuncs::is_integer(dtype()) , "Can only do ^= operator with integer types");
    
    void** s_begin = stride_begin();
    void** s_end = stride_end();
    const void* m_data_ptr = data_ptr();
    auto& mem = this->_mem;
    const int64_t cur_counter = (s_end - s_begin) / 2;
    void* out_ptr_data = this->_mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(_t.dtype()));
    auto& dt = dtype();
    Scalar sc = ptr_to_scalar(_t, m_data_ptr);
    if(sc.isZero()){
       val._t._vals.cexecute_function<WRAP_DTYPES<IntegerTypesL> >([&dt, &cur_counter, &out_ptr_data, &mem, &s_begin, &s_end, &m_data_ptr](auto v_begin, auto v_end){
        using value_t = utils::IteratorBaseType_t<decltype(v_begin)>;
        int64_t counter = 0;
        value_t* v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
        for(;v_begin != v_end; ++v_begin, ++s_begin){
            value_t n_val = (*reinterpret_cast<const value_t*>(*s_begin)) ^ *v_begin;
            if(n_val == *reinterpret_cast<value_t*>(*s_begin)){continue;}
            if(n_val == 0){*s_begin = const_cast<void*>(m_data_ptr); continue;}
            if(*s_begin == m_data_ptr){
                *v_out_data_ptr = n_val;
                ++counter;
                *s_begin = v_out_data_ptr;
                if(counter == cur_counter){
                    out_ptr_data = mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(dt));
                    v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
                    counter = 0;
                }
                continue;
            }

            *reinterpret_cast<value_t*>(*s_begin) = n_val;
        }
        });
        return *this;
 
    }

    val._t._vals.cexecute_function<WRAP_DTYPES<IntegerTypesL> >([&dt, &cur_counter, &out_ptr_data, &mem, &s_begin, &s_end, &m_data_ptr](auto v_begin, auto v_end){
        using value_t = utils::IteratorBaseType_t<decltype(v_begin)>;
        int64_t counter = 0;
        value_t* v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
        for(;v_begin != v_end; ++v_begin, ++s_begin){
            value_t n_val = (*reinterpret_cast<const value_t*>(*s_begin)) ^ *v_begin;
            if(n_val == *reinterpret_cast<value_t*>(*s_begin)){continue;}
            if(*s_begin == m_data_ptr){
                *v_out_data_ptr = n_val;
                ++counter;
                *s_begin = v_out_data_ptr;
                if(counter == cur_counter){
                    out_ptr_data = mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(dt));
                    v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
                    counter = 0;
                }
                continue;
            }

            *reinterpret_cast<value_t*>(*s_begin) = n_val;
        }
    });
    return *this;
}


SparseTensor& SparseTensor::operator ^= (const Tensor& val){
    utils::THROW_EXCEPTION(val.dtype() == dtype(), "Expected dtypes of input ($) and this tensor ($) to match",
                           val.dtype(), dtype());
    utils::THROW_EXCEPTION(val.shape() == shape(), "Expected shapes of input ($) and this tensor ($) to match",
                           val.shape(), shape());
    utils::THROW_EXCEPTION(DTypeFuncs::is_integer(dtype()) , "Can only do ^= operator with integer types");
    
    void** s_begin = stride_begin();
    void** s_end = stride_end();
    const void* m_data_ptr = data_ptr();
    auto& mem = this->_mem;
    const int64_t cur_counter = (s_end - s_begin) / 2;
    void* out_ptr_data = this->_mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(_t.dtype()));
    auto& dt = dtype();
    Scalar sc = ptr_to_scalar(_t, m_data_ptr);
    if(sc.isZero()){
       val._vals.cexecute_function<WRAP_DTYPES<IntegerTypesL> >([&dt, &cur_counter, &out_ptr_data, &mem, &s_begin, &s_end, &m_data_ptr](auto v_begin, auto v_end){
        using value_t = utils::IteratorBaseType_t<decltype(v_begin)>;
        int64_t counter = 0;
        value_t* v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
        for(;v_begin != v_end; ++v_begin, ++s_begin){
            value_t n_val = (*reinterpret_cast<const value_t*>(*s_begin)) ^ *v_begin;
            if(n_val == *reinterpret_cast<value_t*>(*s_begin)){continue;}
            if(n_val == 0){*s_begin = const_cast<void*>(m_data_ptr); continue;}
            if(*s_begin == m_data_ptr){
                *v_out_data_ptr = n_val;
                ++counter;
                *s_begin = v_out_data_ptr;
                if(counter == cur_counter){
                    out_ptr_data = mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(dt));
                    v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
                    counter = 0;
                }
                continue;
            }

            *reinterpret_cast<value_t*>(*s_begin) = n_val;
        }
        });
        return *this;
 
    }
    val._vals.cexecute_function<WRAP_DTYPES<IntegerTypesL> >([&dt, &cur_counter, &out_ptr_data, &mem, &s_begin, &s_end, &m_data_ptr](auto v_begin, auto v_end){
        using value_t = utils::IteratorBaseType_t<decltype(v_begin)>;
        int64_t counter = 0;
        value_t* v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
        for(;v_begin != v_end; ++v_begin, ++s_begin){
            value_t n_val = (*reinterpret_cast<const value_t*>(*s_begin)) ^ *v_begin;
            if(n_val == *reinterpret_cast<value_t*>(*s_begin)){continue;}
            if(*s_begin == m_data_ptr){
                *v_out_data_ptr = n_val;
                ++counter;
                *s_begin = v_out_data_ptr;
                if(counter == cur_counter){
                    out_ptr_data = mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(dt));
                    v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
                    counter = 0;
                }
                continue;
            }

            *reinterpret_cast<value_t*>(*s_begin) = n_val;
        }
    });
    return *this;
}

// SparseTensor& SparseTensor::operator += (const SparseTensor& val){
//     utils::THROW_EXCEPTION(val.dtype() == dtype(), "Expected dtypes of input ($) and this tensor ($) to match",
//                            val.dtype(), dtype());
//     utils::THROW_EXCEPTION(val.shape() == shape(), "Expected shapes of input ($) and this tensor ($) to match",
//                            val.shape(), shape());
//     utils::THROW_EXCEPTION(DTypeFuncs::is_number(dtype()) , "Can only do ^= operator with integer types");
    
//     void** s_begin = stride_begin();
//     void** s_end = stride_end();
//     const void* m_data_ptr = data_ptr();
//     auto& mem = this->_mem;
//     const int64_t cur_counter = (s_end - s_begin) / 2;
//     void* out_ptr_data = this->_mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(_t.dtype()));
//     auto& dt = dtype();
//     Scalar sc = ptr_to_scalar(_t, m_data_ptr);
//     if(sc.isZero()){
//        val._t._vals.cexecute_function<WRAP_DTYPES<NumberTypesL> >([&dt, &cur_counter, &out_ptr_data, &mem, &s_begin, &s_end, &m_data_ptr](auto v_begin, auto v_end){
//         using value_t = utils::IteratorBaseType_t<decltype(v_begin)>;
//         int64_t counter = 0;
//         value_t* v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
//         for(;v_begin != v_end; ++v_begin, ++s_begin){
//             value_t n_val = (*reinterpret_cast<const value_t*>(*s_begin)) ^ *v_begin;
//             if(n_val == *reinterpret_cast<value_t*>(*s_begin)){continue;}
//             if(n_val == 0){*s_begin = const_cast<void*>(m_data_ptr); continue;}
//             if(*s_begin == m_data_ptr){
//                 *v_out_data_ptr = n_val;
//                 ++counter;
//                 *s_begin = v_out_data_ptr;
//                 if(counter == cur_counter){
//                     out_ptr_data = mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(dt));
//                     v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
//                     counter = 0;
//                 }
//                 continue;
//             }

//             *reinterpret_cast<value_t*>(*s_begin) = n_val;
//         }
//         });
//         return *this;
 
//     }

//     val._t._vals.cexecute_function<WRAP_DTYPES<NumberTypesL> >([&dt, &cur_counter, &out_ptr_data, &mem, &s_begin, &s_end, &m_data_ptr, &sc](auto v_begin, auto v_end){
//         using value_t = utils::IteratorBaseType_t<decltype(v_begin)>;
//         int64_t counter = 0;
//         value_t* v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
//         value_t sc_val = sc.to<value_t>();
//         for(;v_begin != v_end; ++v_begin, ++s_begin){
//             value_t n_val = (*reinterpret_cast<const value_t*>(*s_begin)) + *v_begin;
//             // if(n_val == *reinterpret_cast<value_t*>(*s_begin)){continue;}
//             if(n_val == sc_val){*s_begin = const_cast<void*>(m_data_ptr); continue;}
//             if(*s_begin == m_data_ptr){
//                 *v_out_data_ptr = n_val;
//                 ++counter;
//                 *s_begin = v_out_data_ptr;
//                 if(counter == cur_counter){
//                     out_ptr_data = mem->allocate((cur_counter+1) * DTypeFuncs::size_of_dtype(dt));
//                     v_out_data_ptr = reinterpret_cast<value_t*>(out_ptr_data);
//                     counter = 0;
//                 }
//                 continue;
//             }

//             *reinterpret_cast<value_t*>(*s_begin) = n_val;
//         }
//     });
//     return *this;
// }


//this takes a begin and end to a stride
//it checks to see of that strided memory, 
//  how much of it needs to be allocated
//  on the memory pool
//
//it allocates the needed strides to the memory pool, and replaces those indices
// void SparseTensor::garbage_new_memory(void** begin, void** end){
//     int64_t counter = 0;
//     void** b_cpy = begin;
//     for(;begin != end; ++begin){
//         // std::cout << "is in sparse memory: "<< std::boolalpha << this->in_sparse_memory(*begin) << std::noboolalpha << std::endl;
//         counter += this->in_sparse_memory(*begin) ? 1 : 0;
//     }
    
//     if(counter == 0){return;}
//     begin = b_cpy;
//     int64_t size = DTypeFuncs::size_of_dtype(_t.dtype());
//     void* n_ptr = this->_mem->allocate(size * counter);
//     char* ptr = reinterpret_cast<char*>(n_ptr);

//     if(counter == (end - begin)){
//         for(;begin != end; ++begin, ptr += size){
//             *begin = ptr;
//             utils::throw_exception(!this->in_sparse_memory(*begin), "Issue when setting begin, is still in sparse memory?");
//         }
//         return;
//     }
//     for(;begin != end; ++begin){
//         if(this->in_sparse_memory(*begin)){
//             *begin = ptr;
//             utils::throw_exception(!this->in_sparse_memory(*begin), "Issue when setting begin (2), is still in sparse memory?");
//             ptr += size;
//         }
//     }
// }




// SparseTensor& SparseTensor::operator=(Scalar scalar){
//     this->garbage_new_memory(stride_begin(), stride_end());
//     void** begin = stride_begin();
//     void** end = stride_end();
//     _t._vals.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>> >([&begin, &end, &scalar](auto m_begin, auto m_end){
//         using value_t = utils::IteratorBaseType_t<decltype(m_begin)>;
//         value_t val = scalar.to<value_t>();
//         for(;begin != end; ++begin){
//             value_t* p = reinterpret_cast<value_t*>(*begin);
//             *p = val;
//         }
//     });
//     return *this;
// }

// SparseTensor& SparseTensor::operator=(Tensor t){
//     std::cout << "garbaging memory..."<<std::endl;
//     this->garbage_new_memory(stride_begin(), stride_end());
//     std::cout << "garbaged"<<std::endl;
//     this->_t.set_(t);
//     // utils::throw_exception(t.shape() == shape(), "Expected to set tensors of the same shape but got $ = $", shape(), t.shape());
//     // utils::throw_exception(t.dtype() == _t.dtype(), "Expected to set tensors of the same dtype but got $ = $", _t.dtype(), t.dtype());
//     // void** s_begin = stride_begin();
//     // void** s_end = stride_end();
//     // std::cout << "end-begin: " << s_end - s_begin << std::endl;
//     // t._vals.execute_function<WRAP_DTYPES<NumberTypesL, DTypeEnum<DType::Bool>> >([&](auto m_begin, auto m_end, auto begin){
//     //     std::cout << "executing function..."<<std::endl;
//     //     using value_t = utils::IteratorBaseType_t<decltype(m_begin)>;
//     //     std::cout << "got value_t"<<std::endl;
//     //     // BucketIterator_list<value_t> begin(reinterpret_cast<value_t**>(s_begin));
//     //     // BucketIterator_list<value_t> end(reinterpret_cast<value_t**>(s_end));
//     //     if constexpr (utils::iterator_is_contiguous_v<decltype(begin)>){
//     //         std::cout << "got a contiguous iterator"<<std::endl;
//     //     }
//     //     else if constexpr(utils::iterator_is_blocked_v<decltype(begin)>){
//     //         std::cout << "got a blocked iterator"<<std::endl;
//     //     }else if constexpr (utils::iterator_is_list_v<decltype(begin)>){
//     //         std::cout << "got a list iterator"<<std::endl;
//     //         std::cout << "are the iterators equal: "<<std::endl;
//     //         BucketIterator_list<value_t> cpy_begin(reinterpret_cast<value_t**>(s_begin));
//     //         std::cout << std::boolalpha << (cpy_begin == begin) << std::noboolalpha << std::endl;
//     //         value_t* p = static_cast<value_t*>(s_begin[0]);
//     //         std::cout << "holding pointer"<<std::endl;
//     //         std::cout << std::boolalpha << (p == nullptr) << ',' << this->in_sparse_memory(s_begin[0]) << std::noboolalpha << std::endl;
//     //         for(;m_begin != m_end; ++begin, ++m_begin){
//     //             *begin = *m_begin;
//     //             // value_t* p = static_cast<value_t*>(begin[0]);
//     //             // (*p) = (*m_begin);
//     //         }
//     //     }else{
//     //         std::cout << "unknown iterator???"<<std::endl;
//     //     }
//     // }, _t._vals);
//     // std::cout << "finished"<<std::endl;

//     return *this;
// }


}

