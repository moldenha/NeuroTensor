#include "../../Tensor.h"
#include "../../dtype/ArrayVoid.h"
#include "exceptions.hpp"
#include "../../mp/Threading.h"

namespace nt{
namespace functional{



Tensor undilate_(const Tensor& input, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    using size_value_t = Tensor::size_value_t;
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1)) {
        return input;
    }
    utils::throw_exception(input.dims() >= 2, "Expected dim size to be greater than or equal to 2 for undilation but got $", input.dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $}",
                           row_dil, col_dil);
    utils::throw_exception(!input.is_null(),
                           "Cannot undilate a null tensor");

    std::vector<size_value_t> vec = input.shape().Vec();
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;
    SizeRef outp_shape(std::move(vec));
    
    ArrayVoid cpy1 = input.arr_void().bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();


    auto& sh = input.shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t batches = input.numel() / matrix_size;
    for(int64_t b = 0; b < batches; ++b, my_strides += matrix_size){
        void **cur_begin = my_strides;
        for(int64_t r = 0; r < original_rows; r += row_dil, cur_begin += (original_cols * row_dil)){
            void **mBegin = cur_begin;
            for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                *outp_strides++ = *mBegin;
            }
        }
    }

    return Tensor(std::move(cpy), std::move(outp_shape)).set_mutability(input.is_mutable());
}

Tensor undilate_(const Tensor& input, Tensor::size_value_t dil){
    return undilate_(input, 1, dil);
}

Tensor undilate_(const Tensor& input, Tensor::size_value_t row_dil, Tensor::size_value_t col_dil, Tensor::size_value_t chan_dil){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(input);
    using size_value_t = Tensor::size_value_t;
    if ((row_dil == 0 || row_dil == 1) && (col_dil == 0 || col_dil == 1) && (chan_dil == 0 || chan_dil == 1)) {
        return input;
    }

    // Check dimensionality and validate dilation values
    utils::throw_exception(input.dims() >= 3, "Expected dim size to be greater than or equal to 3 for 3D undilation but got $", input.dims());
    utils::throw_exception(row_dil >= 1 && col_dil >= 1 && chan_dil >= 1,
                           "Cannot dilate less than 1 but got dilations of {$, $, $}",
                           row_dil, col_dil, chan_dil);
    utils::throw_exception(!input.is_null(),
                           "Cannot undilate a null tensor");

    // Calculate the new shape after undilation for each dimension
    std::vector<size_value_t> vec = input.shape().Vec();
    vec[vec.size() - 3] = (vec[vec.size() - 3] + (chan_dil - 1)) / chan_dil; // Channel dimension
    vec[vec.size() - 2] = (vec[vec.size() - 2] + (row_dil - 1)) / row_dil;    // Row dimension
    vec.back() = (vec.back() + (col_dil - 1)) / col_dil;                        // Column dimension
    SizeRef outp_shape(std::move(vec));
    
    ArrayVoid cpy1 = input.arr_void().bucket_all_indices();
    ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    void **my_strides = cpy1.stride_begin();
    void **outp_strides = cpy.stride_begin();


    auto& sh = input.shape();
    size_value_t original_cols = sh.back();
    size_value_t original_rows = sh[-2];
    size_value_t original_channels = sh[-3];
    size_value_t matrix_size = original_rows * original_cols;
    size_value_t channel_size = matrix_size * original_channels;
    size_value_t batches = input.numel() / channel_size;

    for(int64_t b = 0; b < batches; ++b, my_strides += channel_size){
        void **cur_begin = my_strides;
        for(int64_t d = 0; d < original_channels; d += chan_dil, cur_begin += matrix_size * chan_dil){
            void **dBegin = cur_begin;
            for(int64_t r = 0; r < original_rows; r += row_dil, dBegin += (original_cols * row_dil)){
                void **mBegin = dBegin;
                for(int64_t c = 0; c < original_cols; c += col_dil, mBegin += col_dil){
                    *outp_strides++ = *mBegin;
                }
            }
        }
    }

    return Tensor(std::move(cpy), std::move(outp_shape)).set_mutability(input.is_mutable());
    
}

// Tensor undilate_(size_value_t dil) const {
    // return this->undilate_(dil, dil);
    // if (dil == 0) {
    //     return contiguous();
    // }

    // // Calculate the original shape before dilation
    // std::vector<size_value_t> vec = shape().Vec();
    // vec.back() = (vec.back() + (dil - 1)) / dil;
    // vec[vec.size() - 2] = (vec[vec.size() - 2] + (dil - 1)) / dil;
    // SizeRef outp_shape(std::move(vec));
    // /* Tensor outp = functional::zeros(SizeRef(vec), dtype); */

    // ArrayVoid cpy1 = _vals.bucket_all_indices();
    // ArrayVoid cpy = cpy1.new_strides(outp_shape.multiply());
    // void **my_strides = cpy1.stride_begin();
    // void **outp_strides = cpy.stride_begin();

    // size_value_t cols = shape()[-1];
    // size_value_t i_total = shape().multiply(-2);
    // size_value_t outp_cols = vec.back();
    // size_value_t outp_i_total = vec[vec.size() - 2];

    // for (size_value_t i = 0; i < numel(); ++i, ++my_strides) {
    //     // Check if the current element should be part of the original tensor
    //     if ((i % (outp_cols * dil)) % dil == 0 &&
    //         (i / (outp_cols * dil)) % dil == 0) {
    //         *outp_strides = *my_strides;
    //         ++outp_strides;
    //     }
    // }

    // return Tensor(std::move(cpy), std::move(outp_shape));
// }


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
