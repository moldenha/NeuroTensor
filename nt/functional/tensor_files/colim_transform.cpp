#include "colim_transform.h"
#include "../cpu/colim_transform.h"
#include "../../Tensor.h"
#include "fill.h"


namespace nt{
namespace functional{

Tensor unfold(const Tensor& x, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride, bool transpose_out){
	utils::throw_exception(x.dims() >= 3, "Expected input tensot to unfold to have dims greater than or equal to 3 but got $D", x.dims());

	const int64_t LKern = kernel_size[0] * kernel_size[1];

	const int64_t L_r = ((x.shape()[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((x.shape()[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;

	utils::throw_exception(L > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", x.shape()[-2], x.shape()[-1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	Tensor im = (x.dims() == 3) ? x.view(1, x.shape()[0], x.shape()[1], x.shape()[2]) : x.flatten(0, -4);
	const int64_t& channels = im.shape()[1];

	Tensor col = zeros({im.shape()[0], im.shape()[1] * LKern, L});
    cpu::unfold2d_(im.arr_void(), col.arr_void(), channels, x.shape()[-2], x.shape()[-1], kernel_size[0], kernel_size[1],
              stride[0], stride[1],
              padding[0], padding[1],
              dilation[0], dilation[1],
              im.shape()[0], transpose_out);
    if(transpose_out) return std::move(col);
	return col.view(col.shape().transpose(-1,-2));

}



Tensor unfold_backward(const Tensor& x, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BROWS = output_size[0];
	const int64_t& BCOLS = output_size[1];
	const int64_t L_r = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(2)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[2]);
	Tensor output = zeros(output_shape, x.dtype);
    cpu::unfold2d_backward_(output.arr_void(), Z.arr_void(),
            channels, BROWS, BCOLS,
			kernel_size[0], kernel_size[1],
			stride[0], stride[1],
			padding[0], padding[1],
			dilation[0], dilation[1],
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return std::move(output);
}

Tensor& unfold_backward(const Tensor& x, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BROWS = output_size[0];
	const int64_t& BCOLS = output_size[1];
	const int64_t L_r = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_c = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L = L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception((L_r*L_c) > 0, "Given input with spatial size ($, $), kernel_size=($, $), dilation=($, $), padding=($, $), calculated shape of the array of sliding blocks as ($, $), but its components must be at least one.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], L_r, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $), kernel_size=($, $), dilation=($, $), padding=($, $), stride=($, $), expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ = $, but got input.size(2)=$.", output_size[0], output_size[1], kernel_size[0], kernel_size[1], dilation[0], dilation[1], padding[0], padding[1], stride[0], stride[1], L_r, L_c, L, Z.shape()[2]);
	utils::throw_exception(output.dtype == x.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, x.dtype);
	utils::throw_exception(output.shape().multiply() == output_shape.multiply(), "Expected to get same shape for output for unfold backward of $ but got $", output_shape, output.shape());
    cpu::unfold2d_backward_(output.arr_void(), Z.arr_void(),
            channels, BROWS, BCOLS,
			kernel_size[0], kernel_size[1],
			stride[0], stride[1],
			padding[0], padding[1],
			dilation[0], dilation[1],
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return output;
}





Tensor fold(const Tensor& x, utils::my_tuple output_size, utils::my_tuple kernel_size, utils::my_tuple dilation, utils::my_tuple padding, utils::my_tuple stride){
    utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $D", x.dims());
    const int64_t& BROWS = output_size[0];
    const int64_t& BCOLS = output_size[1];
    const int64_t L_r = ((BROWS + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_c = ((BCOLS + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L = L_r * L_c;
    const int64_t LKern = kernel_size[0] * kernel_size[1];
    const int64_t batches = (x.dims() == 3) ? x.shape()[0] : 1;
    Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;


    utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
    utils::throw_exception(Z.shape()[2] == L, "Given output_size = $, kernel_size = $, dilation = $, padding = $, stride = $, expected size of input's dimension $ to match the calculated number of sliding blocks $ * $ = $ but got input.shape()[$] = $",
			output_size, kernel_size, dilation, padding, stride, (x.dims() == 3) ? 2 : 1, L_r, L_c, L, (x.dims() == 3) ? 2 : 1, Z.shape()[2]);
    
    const int64_t channels = Z.shape()[1] / LKern;
    std::vector<int64_t> output_shape = {batches, channels, BROWS, BCOLS};
    Tensor output = zeros(output_shape, x.dtype);

    cpu::fold2d_(Z.arr_void(), output.arr_void(),
		    L, L_r, L_c,
		    BROWS, BCOLS,
		    kernel_size[0], kernel_size[1],
		    stride[0], stride[1],
		    dilation[0], dilation[1],
		    padding[0], padding[1],
		    batches, channels);
    return std::move(output);
}



Tensor fold_backward(const Tensor& grad_output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride){
    utils::throw_exception(grad_output.dims() == 4 || grad_output.dims() == 3, "Expected to get a shape with a dimensionality of 4D or 3D, but got $D for fold backward", grad_output.dims());
    const int64_t& BROWS = output_size[0];
    const int64_t& BCOLS = output_size[1];
    const int64_t L_r = ((BROWS + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_c = ((BCOLS + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L = L_r * L_c;
    const int64_t LKern = kernel_size[0] * kernel_size[1];
    const int64_t batches = (grad_output.dims() == 4) ? grad_output.shape()[0] : 1;
    const int64_t channels = grad_output.shape()[-3];
    Tensor Z = (grad_output.dims() == 3) ? grad_output.unsqueeze(0) : grad_output;
    utils::throw_exception(Z.shape()[-1] == BROWS && Z.shape()[-2] == BCOLS, "Expected last dimensions of grad output for fold backward to match output size $ but got ($,$)", output_size, Z.shape()[-2], Z.shape()[-1]);

    /* std::vector<int64_t> output_shape = {batches, channels, BROWS, BCOLS}; */
    SizeRef output_shape({batches, channels * LKern, L});
    Tensor output = zeros(output_shape, grad_output.dtype);


    cpu::fold2d_backward_(output.arr_void(), Z.arr_void(),
		    L, L_r, L_c,
		    BROWS, BCOLS,
		    kernel_size[0], kernel_size[1],
		    stride[0], stride[1],
		    dilation[0], dilation[1],
		    padding[0], padding[1],
		    batches, channels);
    return std::move(output);
}

Tensor& fold_backward(const Tensor& grad_output, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride){
    utils::throw_exception(grad_output.dims() == 4 || grad_output.dims() == 3, "Expected to get a shape with a dimensionality of 4D or 3D, but got $D for fold backward", grad_output.dims());
    const int64_t& BROWS = output_size[0];
    const int64_t& BCOLS = output_size[1];
    const int64_t L_r = ((BROWS + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_c = ((BCOLS + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L = L_r * L_c;
    const int64_t LKern = kernel_size[0] * kernel_size[1];
    const int64_t batches = (grad_output.dims() == 4) ? grad_output.shape()[0] : 1;
    const int64_t channels = grad_output.shape()[-3];
    Tensor Z = (grad_output.dims() == 3) ? grad_output.unsqueeze(0) : grad_output;
    utils::throw_exception(Z.shape()[-1] == BROWS && Z.shape()[-2] == BCOLS, "Expected last dimensions of grad output for fold backward to match output size $ but got ($,$)", output_size, Z.shape()[-2], Z.shape()[-1]);

    /* std::vector<int64_t> output_shape = {batches, channels, BROWS, BCOLS}; */
    SizeRef output_shape({batches, channels * LKern, L});
    //make sure the amount of elements are the same
    utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for fold backward", output_shape, output.shape());
    utils::throw_exception(output.dtype == grad_output.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, grad_output.dtype);


    cpu::fold2d_backward_(output.arr_void(), Z.arr_void(),
		    L, L_r, L_c,
		    BROWS, BCOLS,
		    kernel_size[0], kernel_size[1],
		    stride[0], stride[1],
		    dilation[0], dilation[1],
		    padding[0], padding[1],
		    batches, channels);
    return output;
}





Tensor unfold3d(const Tensor& x, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> stride, bool transpose_out) {
    utils::throw_exception(x.dims() >= 4, "Expected input tensor to unfold to have dims greater than or equal to 4 but got $D", x.dims());

    const int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];

    const int64_t L_d = ((x.shape()[-3] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
    const int64_t L_r = ((x.shape()[-2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
    const int64_t L_c = ((x.shape()[-1] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;
    const int64_t L = L_d * L_r * L_c;

    utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.",
                           x.shape()[-3], x.shape()[-2], x.shape()[-1], 
                           kernel_size[0], kernel_size[1], kernel_size[2], 
                           dilation[0], dilation[1], dilation[2], 
                           padding[0], padding[1], padding[2], 
                           L_d, L_r, L_c);

    Tensor im = (x.dims() == 4) ? x.view(1, x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]) : x.flatten(0, -5);

    const int64_t& channels = im.shape()[1];
    Tensor col = zeros({im.shape()[0], im.shape()[1] * LKern, L});
    cpu::unfold3d_(im.arr_void(), col.arr_void(), channels,
                                   x.shape()[-3], x.shape()[-2], x.shape()[-1],
                                   kernel_size[0], kernel_size[1], kernel_size[2],
                                   stride[0], stride[1], stride[2],
                                   padding[0], padding[1], padding[2],
                                   dilation[0], dilation[1], dilation[2],
                                   im.shape()[0]);

    if (!transpose_out)
        col.RowColSwap();

    return std::move(col);
}


Tensor unfold3d_backward(const Tensor& x, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, const bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BDEPTH = output_size[0];
	const int64_t& BROWS = output_size[1];
	const int64_t& BCOLS = output_size[2];
	const int64_t L_d = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_r = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L_c = ((output_size[2] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;

	const int64_t L = L_d * L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.",
                           x.shape()[-3], x.shape()[-2], x.shape()[-1], 
                           kernel_size[0], kernel_size[1], kernel_size[2], 
                           dilation[0], dilation[1], dilation[2], 
                           padding[0], padding[1], padding[2], 
                           L_d, L_r, L_c);




	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], kernel_size[2], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BDEPTH, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), stride=($, $, $), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ * $ = $, but got input.size(2)=$.", 
			output_size[0], output_size[1], output_size[2], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			stride[0], stride[1], stride[2],
			(transpose_out) ? "true" : "false",
			L_d, L_r, L_c, L, Z.shape()[1]);
	
	Tensor output = zeros(output_shape, x.dtype);
    cpu::unfold3d_backward_(output.arr_void(), 
			Z.arr_void(),  channels,
			BDEPTH, BROWS, BCOLS,
			kernel_size[0], kernel_size[1], kernel_size[2],
			stride[0], stride[1], stride[2],
			padding[0], padding[1], padding[2],
			dilation[0], dilation[1], dilation[2],
			batches);

	
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return std::move(output);
}


Tensor& unfold3d_backward(const Tensor& x, Tensor& output, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, const bool transpose_out){

	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BDEPTH = output_size[0];
	const int64_t& BROWS = output_size[1];
	const int64_t& BCOLS = output_size[2];
	const int64_t L_d = ((output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1;
	const int64_t L_r = ((output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1;
	const int64_t L_c = ((output_size[2] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2]) + 1;

	const int64_t L = L_d * L_r * L_c;
	const int64_t LKern = kernel_size[0] * kernel_size[1] * kernel_size[2];
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L > 0, "Given input with spatial size ($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), calculated shape of the array of sliding blocks as ($, $, $), but its components must be at least one.",
                           x.shape()[-3], x.shape()[-2], x.shape()[-1], 
                           kernel_size[0], kernel_size[1], kernel_size[2], 
                           dilation[0], dilation[1], dilation[2], 
                           padding[0], padding[1], padding[2], 
                           L_d, L_r, L_c);




	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $, $ * $ * $ = $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, kernel_size[0], kernel_size[1], kernel_size[2], LKern, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BDEPTH, BROWS, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($, $, $), kernel_size=($, $, $), dilation=($, $, $), padding=($, $, $), stride=($, $, $), and transpose_out = $, expected size of input's dimension 2 to match the calculated number of sliding blocks $ * $ * $ = $, but got input.size(2)=$.", 
			output_size[0], output_size[1], output_size[2], 
			kernel_size[0], kernel_size[1], kernel_size[2],
			dilation[0], dilation[1], dilation[2],
			padding[0], padding[1], padding[2],
			stride[0], stride[1], stride[2],
			(transpose_out) ? "true" : "false",
			L_d, L_r, L_c, L, Z.shape()[1]);
	
	
	utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for fold backward", output_shape, output.shape());
	utils::throw_exception(output.dtype == x.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, x.dtype);
    cpu::unfold3d_backward_(output.arr_void(), 
			Z.arr_void(), channels,
			BDEPTH, BROWS, BCOLS,
			kernel_size[0], kernel_size[1], kernel_size[2],
			stride[0], stride[1], stride[2],
			padding[0], padding[1], padding[2],
			dilation[0], dilation[1], dilation[2],
			batches);

	
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return output;
}



Tensor unfold1d(const Tensor& x, int64_t kernel_size, int64_t dilation, int64_t padding, int64_t stride, bool transpose_out){
	utils::throw_exception(x.dims() >= 2, "Expected input tensor to unfold to have dims greater than or equal to 2 but got $D", x.dims());

	const int64_t LKern = kernel_size;

	const int64_t L = ((x.shape()[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

	utils::throw_exception(L > 0, "Given input with width ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", x.shape()[-1], kernel_size, dilation, padding, L);
	Tensor im = (x.dims() == 2) ? x.view(1, x.shape()[0], x.shape()[1]) : x.flatten(0, -3);

	Tensor col = zeros({im.shape()[0], im.shape()[1] * LKern, L});
	const int64_t& channels = im.shape()[1];
    cpu::unfold1d_(im.arr_void(), col.arr_void(), channels,
			x.shape()[-1],
			kernel_size,
			stride,
			padding,
			dilation,
			im.shape()[0]);
	if(!transpose_out)
		col.RowColSwap();
	return std::move(col);
}


Tensor unfold1d_backward(const Tensor& x, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BCOLS = output_size;
	const int64_t L_c = ((output_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
	const int64_t& L = L_c;
	const int64_t& LKern = kernel_size;
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L_c > 0, "Given input with spatial size ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", output_size, kernel_size, dilation, padding, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($), kernel_size=($), dilation=($), padding=($), stride=($), expected size of input's dimension 2 to match the calculated number of sliding blocks $, but got input.size(2)=$.", output_size, kernel_size, dilation, padding, stride, L, Z.shape()[2]);
	Tensor output = zeros(output_shape, x.dtype);
    cpu::unfold1d_backward_(output.arr_void(), 
			Z.arr_void(), channels, BCOLS,
			kernel_size,
			stride,
			padding,
			dilation,
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return std::move(output);
}


Tensor& unfold1d_backward(const Tensor& x, Tensor& output, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out){
	utils::throw_exception(x.dims() == 2 || x.dims() == 3, "Expected to get a shape with a dimensionality of 3D or 2D, but got $ dimensions", x.dims());
	const int64_t& BCOLS = output_size;
	const int64_t L_c = ((output_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
	const int64_t& L = L_c;
	const int64_t& LKern = kernel_size;
	const int64_t batches = x.dims() == 3 ? x.shape()[0] : 1;
	utils::throw_exception(L_c > 0, "Given input with spatial size ($), kernel_size=($), dilation=($), padding=($), calculated shape of the array of sliding blocks as ($), but its components must be at least one.", output_size, kernel_size, dilation, padding, L_c);
	Tensor Z = (x.dims() == 2) ? x.unsqueeze(0) : x;
	if(!transpose_out)
		Z = Z.contiguous().RowColSwap();
	utils::throw_exception(Z.shape()[1] % LKern == 0, "Given kernel size of $ must be able to divide input at dimension $, but got input.shape()[$] = $", kernel_size, (x.dims() == 3) ? 1 : 0, (x.dims() == 3) ? 1 : 0, Z.shape()[1]);
	const int64_t channels = Z.shape()[1] / LKern;
	SizeRef output_shape({batches, channels, BCOLS});
	utils::throw_exception(Z.shape()[2] == L, "Given output_size=($), kernel_size=($), dilation=($), padding=($), stride=($), expected size of input's dimension 2 to match the calculated number of sliding blocks $, but got input.size(2)=$.", output_size, kernel_size, dilation, padding, stride, L, Z.shape()[2]);
	utils::throw_exception(output_shape.multiply() == output.shape().multiply(), "Expected output shape to match $ but got $ for fold backward", output_shape, output.shape());
	utils::throw_exception(output.dtype == x.dtype, "Expected dtypes of grad and output to match for fold backward but got $ for output and $ for the grad", output.dtype, x.dtype);
    cpu::unfold1d_backward_(output.arr_void(), 
			Z.arr_void(), channels, BCOLS,
			kernel_size,
			stride,
			padding,
			dilation,
			batches);
	if(!transpose_out && x.is_contiguous())
		Z.RowColSwap();
	return output;
}



}} //nt::functional
