#include "layers.h"


#include <_types/_uint32_t.h>
#include <sys/types.h>
#include <variant>
#include <tuple>
#include "../dtype/ArrayVoid.hpp"
#include "../functional/functional.h"

#ifdef USE_PARALLEL
	#include <tbb/parallel_for_each.h>
	#include <tbb/parallel_for.h>
	#include <tbb/blocked_range.h>
	#include <tbb/blocked_range2d.h>
	#include <tbb/blocked_range3d.h>
	/* #include <tbb/blocked_rangeNd.h> */
#endif

namespace nt{
namespace layers{


Unfold::Unfold(utils::my_tuple kernel_size, 
			utils::my_tuple dilation,
			utils::my_tuple padding,
			utils::my_tuple stride,
			bool transpose_out)
	:kernel_size(kernel_size), dilation(dilation), padding(padding), stride(stride), out_transpose(transpose_out), BLSHAPE(0), BROWS(0), BCOLS(0)
{
	LKern = kernel_size[0] * kernel_size[1];
}


//while the following works, when it comes to larger tensors it is simply too slow
//going to implement a version that is purely for loops
//should not only be faster, but easier for parallelization
//maybe going to use the pool module that I had created for a previous project, should potentially work better
/* Tensor Unfold::forward(const Tensor& x){ */
/* 	utils::throw_exception(x.dims() == 4, "Expected dimensions of Tensor to unfold to be 4 but got $", x.dims()); */
/* 	uint32_t L = ((x.shape()[-1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1; */ 
/* 	L *= ((x.shape()[-2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1; */
/* 	SizeRef outp_shape_ut({x.shape()[0], L, x.shape()[1] * LKern}); */
/* 	bool dilate = (dilation[0] > 1 || dilation[1] > 1); */
/* 	Tensor X = dilate ? x.dilate_mem_(dilation[0]) : x; */
/* 	if(padding[0] != 0 || padding[1] != 0) */
/* 		X = X.pad({padding[0], padding[1]}); */
	
/* 	Tensor unfolded = X.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1]); */
/* 	uint32_t neg_one = unfolded.numel() / (x.shape()[0] * x.shape()[1] * kernel_size[0] * kernel_size[1]); */
/* 	std::cout << "finished unfold"<<std::endl; */
/* 	unfolded = unfolded.view({x.shape()[0], x.shape()[1], neg_one, kernel_size[0], kernel_size[1]}); */
/* 	std::cout<<"doing permute"<<std::endl; */
/* 	unfolded = unfolded.permute({0, 2, 1, 3, 4}).contiguous().view(outp_shape_ut); */
/* 	std::cout<<"doing row col swap"<<std::endl; */
/* 	/1* unfolded.RowColSwap(); *1/ */
/* 	unfolded = unfolded.transpose(-1,-2); */
/* 	std::cout<<"did row col swap"<<std::endl; */
/* 	/1* unfolded = unfolded.view({unfolded.shape()[0], unfolded.shape()[2], unfolded.shape()[1]}); *1/ */

/* 	return std::move(unfolded); */
/* } */


//%s/const uint32_t& k_r, const uint32_t& k_c, const uint32_t& s_r, const uint32_t& s_c, const uint32_t& d_r, const uint32_t& d_c/const uint32_t\& k_r, const uint32_t\& k_c, const uint32_t\& s_r, cons

/* Tensor im2col_2_dim */

//im2col_nn_layer_2d_T function: 142
//im2col_nn_layer_2d + RowColSwap_contiguous: 221
//im2col_nn_layer_2d + transpose(-1,-2): 180
//im2col_nn_layer_2d + RowColSwap: 168
//clearly the im2col_nn_layer_2d_T function is most efficient 

Tensor Unfold::forward(const Tensor& x){
	BROWS = x.shape()[-2];
	BCOLS = x.shape()[-1];
	return functional::unfold(x, this->kernel_size, this->dilation, this->padding, this->stride, this->out_transpose);

}

Tensor Unfold::backward(const Tensor& dz) const{
	return functional::fold(dz, {BROWS, BCOLS}, this->kernel_size, this->dilation, this->padding, this->stride);
}

Tensor Unfold::eval(const Tensor& x) const {
	return functional::unfold(x, this->kernel_size, this->dilation, this->padding, this->stride, this->out_transpose);
}



Fold::Fold(utils::my_tuple output_size, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride, bool transpose_out)
	:uf(kernel_size, dilation, padding, stride, transpose_out)
{
	int32_t L_r = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	int32_t L_c = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1; 
	uint32_t L = L_r * L_c;
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c); 
	uf.BLSHAPE = L;
	uf.BCOLS = output_size[1];
	uf.BROWS = output_size[0];
}

Tensor Fold::forward(const Tensor &x){
	return uf.backward(x);
}

Tensor Fold::eval(const Tensor& x) const{
	return uf.backward(x);
}

Tensor Fold::backward(const Tensor& dz){
	utils::throw_exception(dz.dims() > 1, "Expected backward dZ dims to be greater than 1 but got $", dz.dims());
	utils::throw_exception(dz.shape()[-1] == uf.BCOLS && dz.shape()[-2] == uf.BROWS, "Expected last 2 dimensions of dZ shape to be ($,$) but instead got ($,$)", uf.BROWS, uf.BCOLS, dz.shape()[-2], dz.shape()[-1]);
	return uf.eval(dz);
}

}
}
