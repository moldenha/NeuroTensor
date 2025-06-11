#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.h"
#include "../../dtype/ArrayVoid.hpp"
#include "exceptions.hpp"
#include "../../mp/Threading.h"

namespace nt{
namespace functional{




//this is for when the stride is already changed, makes certain functions easier
//this is more akin to pytorch's version
//however, purely because of the way that the cat function works, this could be dangerous
//for example if I did cat(A, B[2], C)
//it would look at all of B, not just B[2]
Tensor as_strided_force_contiguity(const Tensor& input, const SizeRef& n_size, const SizeRef& n_stride, const int64_t& storage_offset){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
	utils::THROW_EXCEPTION(n_size.size() == n_stride.size(), "Expected to have same amount of strides as dimensions for as strided");
	//bound force contiguity bucket basically takes the tensor's memory
	//and looks at each bucket of memory (for example if there was a concatenation that happened)
	//and it looks at all instances of memory
	ArrayVoid strided_vals = input.strides() != input.getChangedStrides() ? input.arr_void().bound_force_contiguity_bucket() : (input.arr_void().get_bucket().is_strided()) ? input.arr_void() : input.arr_void().bucket_all_indices(); //make input strided so that memory can be accessed one at a time
	ArrayVoid output = strided_vals.new_strides(n_size.multiply()); //make the output strided but with the new size
	void** in_begin = strided_vals.stride_begin(); //start at the correct indice to begin
	void** out_begin = output.stride_begin();//the starting point of the new tensor
	// Calculate the total number of elements in the new tensor
	Tensor::size_value_t total_elements = n_size.multiply();
	//get contiguous strides of the new size
	std::vector<Tensor::size_value_t> multiplicities = n_size.strides();
	//make a reference to numel (reduce overhead)
	const uint64_t& num = strided_vals.Size();
	// Fill the output tensor with strided values from the input tensor
    threading::preferential_parallel_for(threading::block_ranges<1>(0, total_elements),
    [&](threading::blocked_range<1> block){
    for (Tensor::size_value_t i = block.begin[0]; i < block.end[0]; ++i) {
		Tensor::size_value_t offset = storage_offset;
		Tensor::size_value_t index = i;
			for (Tensor::size_value_t j = 0; j < n_size.size()-1; ++j) {
				Tensor::size_value_t mult = (index / multiplicities[j+1]);
				offset += mult * n_stride[j];
				index -= mult * multiplicities[j+1];
			}
		offset += index * n_stride.back();
		//make sure offset isn't out of range, if so, subtract by input.numel() until it is back in range
		offset = (offset < num) ? offset : offset % num;
		out_begin[i] = in_begin[offset];
	}
 
    }
    );

    	
	if(n_stride.size() == n_size.size()){
		std::vector<Tensor::size_value_t> out_strides = {n_size.multiply()};
		out_strides.insert(out_strides.end(), n_stride.begin(), n_stride.end());
		return Tensor(output, std::move(n_size), out_strides);
	}
	return Tensor(output, std::move(n_size), n_stride.Vec()).set_mutability(input.is_mutable());
}

Tensor diagonal(const Tensor& t, bool keep_dims){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(t);
    if(t.dtype == DType::TensorObj && t.dims() < 2){
        Tensor out = Tensor::makeNullTensorArray(t.numel());
        Tensor* out_begin = reinterpret_cast<Tensor*>(out.data_ptr());
        t.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::TensorObj> > >( [&out_begin, &keep_dims](auto begin, auto end){
            for(;begin != end; ++begin, ++out_begin)
                *out_begin = diagonal(*begin, keep_dims);
        });
        out.set_mutability(t.is_mutable());
        return std::move(out);
    }
    utils::throw_exception(t.dims() >= 2, "Error: Cannot get diagonal of a tensor with less than 2 dims and shape of $", t.shape());
    const int64_t& rows = t.shape()[-2];
    const int64_t& cols = t.shape()[-1];
    const int64_t batches = t.numel() / (rows * cols);
    const int64_t out_cols = std::min(rows, cols);
    const int64_t mat_size = rows * cols;
    ArrayVoid my_vals = t.arr_void().bucket_all_indices();
    ArrayVoid new_vals = t.arr_void().new_strides(batches * out_cols);
    void **out_begin = new_vals.stride_begin();
    void **my_begin = my_vals.stride_begin();
    threading::preferential_parallel_for(
    threading::block_ranges<2>(0, batches, 0, out_cols),
    [&](threading::blocked_range<2> range){
    for(int64_t b = range.begin[0]; b < range.end[0]; ++b){
        for(int64_t r = range.begin[1]; r < range.end[1]; ++r){
            out_begin[b * out_cols + r] = my_begin[(b * mat_size) + r * cols + r];
        }
    }});
    if(keep_dims){
        return Tensor(new_vals,
                      t.shape().redo_index(-1, 1).redo_index(-1, out_cols))
                        .set_mutability(t.is_mutable());
    }
    return Tensor(new_vals,
                  t.shape().delete_index(-1).redo_index(-1, out_cols))
                    .set_mutability(t.is_mutable());
}


//this goes based off a comparison of the original strides inputted
//so based off of input.strides()
//which is basically contiguous viewing
//but it goes based off of the way the strides are already implanted in memory
Tensor as_strided(const Tensor& input, const SizeRef n_size, SizeRef n_stride, const int64_t storage_offset, bool whole_tensor){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
	if(n_stride.size() == n_size.size()+1){n_stride = n_stride.pop_front();}
	if(whole_tensor){return as_strided_force_contiguity(input, n_size, n_stride, storage_offset);}
	utils::THROW_EXCEPTION(n_size.size() == n_stride.size(), "Expected to have same amount of strides as dimensions for as size or one more, where the last dimension represents n_size.multiply()");
	//bound force contiguity bucket basically takes the tensor's memory
	//and looks at each bucket of memory (for example if there was a concatenation that happened)
	//and it looks at all instances of memory
	ArrayVoid strided_vals = (input.arr_void().get_bucket().is_strided()) ? input.arr_void() : input.arr_void().bucket_all_indices(); //make input strided so that memory can be accessed one at a time
	ArrayVoid output = strided_vals.new_strides(n_size.multiply()); //make the output strided but with the new size
	void** in_begin = strided_vals.stride_begin(); //start at the correct indice to begin
	void** out_begin = output.stride_begin();//the starting point of the new tensor
	// Calculate the total number of elements in the new tensor
	Tensor::size_value_t total_elements = n_size.multiply();
	//get contiguous strides of the new size
	std::vector<Tensor::size_value_t> multiplicities = n_size.strides();
	//make a reference to numel (reduce overhead)
	const uint64_t& num = strided_vals.Size();
	// Fill the output tensor with strided values from the input tensor
    threading::preferential_parallel_for(threading::block_ranges<1>(0, total_elements),
    [&](threading::blocked_range<1> block){
	for (Tensor::size_value_t i = block.begin[0]; i < block.end[0]; ++i) {
		Tensor::size_value_t offset = storage_offset;
		Tensor::size_value_t index = i;
			for (Tensor::size_value_t j = 0; j < n_size.size()-1; ++j) {
				Tensor::size_value_t mult = (index / multiplicities[j+1]);
				offset += mult * n_stride[j];
				index -= mult * multiplicities[j+1];
			}
		offset += index * n_stride.back();
		//make sure offset isn't out of range, if so, subtract by input.numel() until it is back in range
		offset = (offset < num) ? offset : offset % num;
		out_begin[i] = in_begin[offset];
	}});
	
	if(n_stride.size() == n_size.size()){
		std::vector<Tensor::size_value_t> out_strides = {n_size.multiply()};
		out_strides.insert(out_strides.end(), n_stride.begin(), n_stride.end());
		return Tensor(output, std::move(n_size), out_strides);
	}
	return Tensor(output, std::move(n_size), n_stride.Vec()).set_mutability(input.is_mutable());
}



}
}
