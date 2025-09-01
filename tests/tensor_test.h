//this is a file for testing the tensor class and some of it's functionality
//needs to be extended
#include <nt/Tensor.h>
#include <nt/dtype/ArrayVoid.hpp>
#include <nt/functional/functional.h>
#include <nt/functional/tensor_files/mesh.h>

#include <iostream>
#include <string>
#include <utility>


void tensor_test_working(){


	nt::Tensor t({3,4,5});
	/* float* ptr = t.data_ptr(); */
	std::cout<<"initialized tensor"<<std::endl;
	std::cout<<"shape: "<<t.shape()<<std::endl;
	std::cout<<"contig count: "<<t.contig_count()<<std::endl;
	float* begin = reinterpret_cast<float*>(t.data_ptr());
	for(uint32_t i = 0; i < t.shape().multiply(); ++i){
		*begin = i;
		++begin;
	}
	std::cout<<"contig count: "<<t.contig_count()<<std::endl;
	std::cout << "printing..."<<std::endl;
	std::cout << t<< std::endl;
	std::cout << "making aa"<<std::endl;
	nt::Tensor a = t[1];
	std::cout << "a numel: "<<a.numel()<<std::endl;
	std::cout << "a arr_void size: "<<a.arr_void().Size()<<std::endl;
	std::cout << a<<std::endl;
	nt::Tensor aa = t[1][2];
	std::cout << "made aa"<<std::endl;
	t.print();
	std::cout << "printed t"<<std::endl;
	std::cout << aa.shape() << std::endl;
	aa.arr_void().cexecute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Float32> > > (
			[](auto abegin, auto aend){
				uint32_t i = 0;
				for(;(abegin + 1) != aend; ++abegin, ++i){
					if(i > 10)
						break;
					std::cout << *abegin << std::endl;
				}
			});
	aa.print();
	std::cout << "printed aa"<<std::endl;

	std::cout << t[1][2].shape() << ", " << aa.shape() << std::endl;
	std::cout << t[1][2] << std::endl; // this now works
	std::cout << aa << std::endl;
	std::cout << t << std::endl;
	nt::Tensor begin_t = t[1][2];	
	std::cout<<"contig count: "<<t.contig_count()<<std::endl;
	std::cout<<"contig count: "<<begin_t.contig_count()<<std::endl;

	begin = reinterpret_cast<float*>(begin_t.data_ptr());
	std::cout<<"contig count: "<<t.contig_count()<<std::endl;
	for(uint32_t i = 0; i < begin_t.shape().multiply(); ++i){
		*begin = i;
		++begin;
	}

	t[2] = 3.1415f;
	std::cout << "t[2] = 3.1415: "<<t<<std::endl;
	t[2].print();
	begin_t = begin_t.contiguous();

	std::cout<<"shape: ("<<t.shape()[0]<<","<<t.shape()[1]<<","<<t.shape()[2]<<")"<<std::endl;
	std::cout<<"contig count: "<<t.contig_count()<<std::endl;
	t.print();
	t = t.view({3,20});
	std::cout<<"contig count: "<<t.contig_count()<<std::endl;
	t.print();
	nt::Tensor t2 = t.contiguous().view({2,3,2,5});
	t.print();
	std::cout<<"contig count: "<<t.contig_count()<<std::endl;
	std::cout<<"t2 contig count: "<<t2.contig_count()<<std::endl;
	t = t.view(60);
	t.print();
	
	std::cout<<"t2:"<<std::endl;
	t2.print();
	t = t.view(3,4,5);
	t.arr_void().iota(0);
	std::cout << "T no transpose: "<<t<<std::endl;
	std::cout << "T transposed (-1,-2): "<<t.transpose(-1,-2)<<std::endl;
	std::cout << "T transposed (0,1): "<<t.transpose(0,1)<<std::endl;
	std::cout << "T transposed (0,2): "<<t.transpose(0,2)<<std::endl;
}


void test_permute(){
	nt::Tensor t = nt::functional::arange(2*3*4*5).view(2,3,4,5);
	std::cout<<"original tensor:"<<std::endl;
	t.print();
	nt::Tensor pt = t.permute({2,1,3,0});
	std::cout<<"permuted:"<<std::endl;
	pt.print();
}

void tensor_div_test(){
	nt::Tensor t = nt::functional::arange(2*3*4*5, nt::DType::Double).view({2,3,4,5});
	nt::Tensor t2 = t.div(20).view({4,5});
	std::cout<<t2<<std::endl;
	std::cout<<t2[2]<<std::endl;
	/* t2[4].item<double>() = 300.90192342; */
	t2[1] = double(300.90192342);
	std::cout<<t2<<std::endl;
	std::cout<<t<<std::endl;
	double val = t[0][1][1][2].item<double>();
	std::cout<<val<<std::endl;
}

void Tensor_Split_Axis(){
	nt::Tensor t = nt::functional::arange(2*3*3*5, nt::DType::Double).view({2,3,3,5});
	t = t.transpose(0,2);
	nt::Tensor t2 = t.split_axis(0);
	/* std::cout<<t2<<std::endl; */
	t2.print();
	

	/* t2.arr_void().iota(0.0); */
	/* std::cout<<"after splitting by cols:"<<std::endl; */
	/* std::cout<<t<<std::endl; */
	/* std::cout<<t2.shape()<<std::endl; */
	/* t2 = t.split_axis(-2); */
	/* t2.arr_void().iota(0.0); */
	/* std::cout<<"after splitting by rows:"<<std::endl; */
	/* std::cout<<t<<std::endl; */
	/* std::cout<<t2.shape()<<std::endl; */
}

void tensor_cat(){
	nt::Tensor t = nt::functional::arange(2*3*4*5, nt::DType::Double).view({2,3,4,5});
	nt::Tensor t2 = nt::functional::arange(2*3*4*5, nt::DType::Double).view({2,3,4,5});
	std::cout << t << std::endl;
	/* nt::Tensor t2 = nt::functional::zeros({2,2,4,5}, nt::DType::Double); */
	/* std::cout<<t2<<std::endl; */
	nt::Tensor x = nt::functional::cat(t, t2, 1);
	std::cout<<x<<std::endl;
}

void permute_save_load(){
	nt::Tensor t = nt::functional::arange(5*4*3*2*6).view({5,4,3,2,6});
	std::cout<<"made t"<<std::endl;
	t = t.permute({1,0,3,2,4});
	std::cout<<"permuted"<<std::endl;
	nt::functional::save(t, "binary_save/tensor_float.nt");
	std::cout<<"wrote"<<std::endl;
	nt::Tensor t2 = nt::functional::load("binary_save/tensor_float.nt");
	std::cout<<t2.shape()<<std::endl;
	std::cout<<t.shape()<<std::endl;
	/* std::cout<<t2<<std::endl; */
	std::cout<<std::boolalpha<<nt::functional::all(t == t2)<<std::noboolalpha<<std::endl;

}

void permute_indexed_test(){
	nt::Tensor t = nt::functional::arange(5*4*3*2*6).view({5,4,3,2,6});
	nt::Tensor t2 = t.permute({1,0,3,2,4});
	std::cout<<"permuted"<<std::endl;
	std::cout<<t2<<std::endl;
	/* std::cout<<"printing strides:"<<std::endl; */
	/* void** strs = t2.arr_void().strides_cbegin(); */
	/* for(uint32_t i = 0; i < t2.numel(); ++i, ++strs){ */
	/* 	std::cout << *reinterpret_cast<float*>(*strs)<<" "; */
	/* } */
	/* std::cout<<std::endl; */
	nt::Tensor t3 = t2[2].permute({2,0,1,3});
	std::cout<<"sub t3:"<<std::endl;
	std::cout<<t3<<std::endl;
	t3[1] = 2.0;
	std::cout<<"t1: "<<std::endl;
	std::cout<< t <<std::endl;
}


void exp_test(){
	nt::Tensor t = nt::functional::arange(3*2*6).view({3,2,6});
	std::cout << t<<std::endl;
	t.exp_();
	std::cout << t<<std::endl;
}

void hadamard_test(){
	nt::Tensor t = nt::functional::arange(3*1*2*1*4*4).view({3,1,2,1,4,4});
	nt::Tensor outp = nt::functional::zeros({3,3,2,2,4,4});
	outp = 1;
	/* std::cout << outp << std::endl; */
	std::cout<<"Post multiplication:"<<std::endl;
	outp *= t;
	std::cout << outp << std::endl;
}

// Function to demonstrate reverse iteration
template <typename Func, std::size_t... Is>
void iterate_reverse(Func func, std::index_sequence<Is...>) {
    (func(Is), ...);  // Fold expression to call func for each index
}

void sum_as_test(){
	nt::Tensor a = nt::functional::randint(0, 6, {3,4,5,6}, nt::DType::Float32);
	std::cout << "A: "<<a<<std::endl;
	nt::Tensor to_sum = nt::functional::randint(0, 6, {3,1,5,1}, nt::DType::Float32);
	std::cout << "to sum: "<<to_sum<<std::endl;
	nt::Tensor out_sum = a.sum_as(to_sum);
	std::cout << "out sum: "<<out_sum << std::endl;
	std::cout << "addition: "<<out_sum + to_sum << std::endl;
	out_sum.arr_void().execute_function<nt::WRAP_DTYPES<nt::DTypeEnum<nt::DType::Float32> > > (
			[](auto abegin, auto aend, auto bbegin){
				for(;abegin != aend; ++abegin, ++bbegin){
					std::cout << *abegin << " + "<<*bbegin << " = "<< (*abegin + *bbegin) << std::endl;
				}
			}, to_sum.arr_void());
	std::cout << "addition: "<<out_sum.clone() + to_sum << std::endl;

	//this function determined the need for reverse_index_sequence, so also testing this
	std::cout << "iterate reverse from 10 test:"<<std::endl;
	iterate_reverse([](std::size_t i) { std::cout << i << " "; }, nt::utils::reverse_index_sequence<10>{});
	std::cout << std::endl << "normal index sequence:"<<std::endl;
	iterate_reverse([](std::size_t i) {std::cout << i << ' ';}, std::make_index_sequence<10>{});
	std::cout << std::endl;
	
	nt::Tensor b_copy = to_sum.clone();
	b_copy += a;
	std::cout << "to_sum += a = "<< b_copy  << std::endl;
	a += to_sum;
	std::cout << "a += to_sum = "<< a << std::endl;
}


void operator_test(){
    nt::Tensor a = nt::functional::randint(0, 6, {1, 5}, nt::DType::Float32);
    nt::Tensor b = nt::functional::randint(0, 6, {3, 5}, nt::DType::Float32);
    std::cout << nt::noprintdtype;
    std::cout << "A: "<<a<<std::endl;
    std::cout << "B: "<<b<<std::endl;
    std::cout << "A - B[0]: "<< a - b[0] << std::endl;
    std::cout << "A - B[1]: "<< a - b[1] << std::endl;
    std::cout << "A - B[2]: "<< a - b[2] << std::endl;
    // std::cout << "B[0]: "<<b[0 ]<< ',' << *reinterpret_cast<float*>(b[0].data_ptr()) << std::endl;
    // std::cout << "B[1]: "<<b[1] << ',' << *reinterpret_cast<float*>(b[1].data_ptr()) << std::endl;
    // std::cout << "B[2]: "<<b[2] << ',' << *reinterpret_cast<float*>(b[2].data_ptr()) << std::endl;

    nt::Tensor s_b = b.sum_as(a);
    std::cout << "s_b: "<<s_b<<std::endl;
    nt::Tensor c = a.clone();
    c -= s_b;
    std::cout << "C -= B: "<< c << std::endl;
    nt::Tensor c_a = a.clone();
    c_a -= b[0];
    c_a -= b[1];
    c_a -= b[2];
    std::cout << "A -= B [individually]: " << c_a << std::endl;

    a -= b;
    std::cout << "A -= B: " << a << std::endl;

}

void operator_multiply_test(){
    nt::Tensor a = nt::functional::randint(0, 6, {1, 5}, nt::DType::Float32);
    nt::Tensor b = nt::functional::randint(0, 6, {3, 5}, nt::DType::Float32);
    std::cout << nt::noprintdtype;
    std::cout << "A: "<<a<<std::endl;
    std::cout << "B: "<<b<<std::endl;
    std::cout << "A * B[0]: "<< a * b[0] << std::endl;
    std::cout << "A * B[1]: "<< a * b[1] << std::endl;
    std::cout << "A * B[2]: "<< a * b[2] << std::endl;
    // std::cout << "B[0]: "<<b[0 ]<< ',' << *reinterpret_cast<float*>(b[0].data_ptr()) << std::endl;
    // std::cout << "B[1]: "<<b[1] << ',' << *reinterpret_cast<float*>(b[1].data_ptr()) << std::endl;
    // std::cout << "B[2]: "<<b[2] << ',' << *reinterpret_cast<float*>(b[2].data_ptr()) << std::endl;

    nt::Tensor c_a = a.clone();
    c_a *= b[0];
    c_a *= b[1];
    c_a *= b[2];
    std::cout << "A *= B [individually]: " << c_a << std::endl;

    //should cause an error
    a *= b;
    std::cout << "A *= B: " << a << std::endl;

}


void conv_test(){
	nt::Tensor a_1d = nt::functional::randn({3,400,500});
	nt::Tensor k_1d = nt::functional::randn({500,400,10});
	std::cout << "convoluting..."<<std::endl;
	nt::Tensor o_1d = nt::functional::conv1d(a_1d, k_1d); //shape should be (3,500,491)
	std::cout << "conv1d output shape: "<<o_1d.shape()<<std::endl;

	nt::Tensor a_2d = nt::functional::randn({2,500,90,90});
	nt::Tensor k_2d = nt::functional::randn({600,500,3,3});
	std::cout << "convoluting..."<<std::endl;
	nt::Tensor o_2d = nt::functional::conv2d(a_2d, k_2d); //shape should be (2,600,88,88)
	std::cout << "conv2d output shape: "<<o_2d.shape()<<std::endl;

	nt::Tensor a_3d = nt::functional::randn({2,500,90,90,90});
	nt::Tensor k_3d = nt::functional::randn({600,500,3,3,3});
	std::cout << "convoluting..."<<std::endl;
	nt::Tensor o_3d = nt::functional::conv3d(a_3d, k_3d); //shape should be (2,600,88,88,88) (also pytorch's is really slow)
	std::cout << "conv3d output shape: "<<o_3d.shape() << std::endl;

	
}

void begin_convT_test(){
    //this is the _strides aspect:
    nt::Tensor x = nt::functional::randint(1, 3, {2,3,10,10}, nt::DType::Float32);
    nt::Tensor w = nt::functional::randn({3, 4, 5, 5}).to(nt::DType::Float32);
    std::cout << x.shape().multiply(-1) << ',' << x.shape().multiply(-2) << std::endl;
    std::vector<nt::Tensor::size_value_t> dils = {2,2,2};
    std::vector<nt::Tensor::size_value_t> multiplies(dils.size());
    for (int i = dils.size() - 1; i >= 1; --i) {
        multiplies[dils.size() - 1 - i] = x.shape().multiply(-1 * i);
    }
    multiplies.back() = 0;
    std::vector<nt::Tensor::size_value_t> back_adds(dils.size());
    for(int i = 0; i < dils.size(); ++i){
        back_adds[i] =  multiplies[i] * (dils[i]-1) + 1; 
    }
    for (int i = 0; i < multiplies.size(); ++i) {
        multiplies[i] = x.shape().multiply(-1 * (i+1));
    }

    for(int i = 0; i < multiplies.size(); ++i){
        std::cout << '('<<multiplies[i] << ',' << back_adds[i] << ')';
    }
    std::cout << std::endl;
    //this would be equivalent of the _stride function:
    int64_t _sh = 1; //stride_rows = 2
    int64_t _sw = 1; //stride_cols = 2
    int64_t _dh = 1; //dilation_rows = 2
    int64_t _dw = 1; //dilation_cols = 2
    int64_t _ph = 1;
    int64_t _pw = 1;

    int64_t kernel_h = 5;
    int64_t kernel_w = 5;
    int64_t effective_h = (kernel_h - 1) * _dh + 1;
    int64_t effective_w = (kernel_w - 1) * _dw + 1;
    // Calculate minimal padding
    int64_t pad_h = (effective_h - 1) + _ph;
    int64_t pad_w = (effective_w - 1) + _pw;
    std::cout << "pad_h: "<<pad_h<<std::endl;
    std::cout << "pad_w: "<<pad_w<<std::endl;
    nt::Tensor _strided_input = (_sh == 1 && _sw == 1) ? x : x.dilate(_sh, _sw); //takes stride into account
    std::cout <<"strided input shape: "<< _strided_input.shape() << std::endl;
    nt::Tensor w_transpose = w.transpose(0,1).flip(-2).flip(-1);
    std::cout << "w_transpose shape: "<<w_transpose.shape()<<std::endl;
    //(tensor, kernel_shape, dilation, padding, stride)
    //padding was kernel -1 + padding
    nt::Tensor col_matrix = nt::functional::unfold(
        _strided_input,               // Input tensor
        {kernel_h, kernel_w},                       // Kernel shape
        {_dh, _dw},                   // Dilation factors
        {pad_h, pad_w}, // Padding to account for dilation
        {1, 1}                        // Stride (set to 1 for transpose convolution)
    );    
        std::cout << "col matrix shape: "<< col_matrix.shape() << std::endl;
    w_transpose = w_transpose.view(w_transpose.shape()[0], -1);
    std::cout << "w_transpose shape: "<<w_transpose.shape() << std::endl;
    nt::Tensor outp_unfold = nt::functional::matmult(w_transpose, col_matrix);
    std::cout << "outp_unfold shape: "<<outp_unfold.shape()<<std::endl;
    // nt::Tensor output = outp_unfold.view(2, 4, 23, 23);
    // std::cout << "output shape: "<<output.shape() << std::endl;
}


void grouped_conv_transpose_test(){
    int64_t groups = 3;
    int64_t out_channels = 3 * 2;
    int64_t in_channels = 2 * 3;
    int64_t batch_size = 4;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t output_padding = 0;
    int64_t dilation = 1;
    nt::Tensor x = nt::functional::randn({batch_size, in_channels, 4});
    nt::Tensor w = nt::functional::randn({in_channels, out_channels / groups, 4});

    nt::Tensor func_output = nt::functional::conv_transpose1d(x, w, stride, padding, output_padding, dilation, groups);
    std::cout << "conv_transpose1d returned tensor of shape: "<<func_output.shape() << std::endl;
    
    int64_t _pad = ((w.shape()[-1] - 1) * dilation - padding);
    nt::Tensor x_unfold = nt::functional::unfold1d(x, w.shape()[-1], dilation, _pad, 1);
    const int64_t Cout = (x.shape()[-1] - 1) * stride - 2 * padding + dilation * (x.shape()[-1] - 1) + 1 + output_padding;
    
    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1]);
    int64_t k_add = w.shape()[0] / groups;
    std::cout << "k_add is: "<<k_add<<std::endl;
    nt::Tensor x_parts = x_unfold.split_axis({nt::range_(0, x_unfold.shape()[0]), nt::range_(0, add)});
    nt::Tensor w_transpose = w.transpose(0, 1).flip(-1);
    std::cout << "w_transpose shape: "<<w_transpose.shape() << std::endl;
    nt::Tensor k_parts = w_transpose.split_axis({nt::range_(0, w_transpose.shape()[0]), nt::range_(0, k_add)}).view_Tensors(k_add, -1);
    std::cout << "got k_parts"<<std::endl;
    nt::Tensor output = nt::functional::matmult(k_parts, x_parts);
    int64_t per_row = output.numel() / (groups * Cout * batch_size);
    std::cout << "matmult output shape: "<<output.shape() << std::endl;
    output = output.view(groups, -1, per_row, Cout).transpose(0,1).view(x.shape()[0], -1, Cout).contiguous();
    std::cout << "output shape: "<<output.shape() << std::endl;

    std::cout << "func output:" << func_output<<std::endl;
    std::cout << "grouped matmult output: "<<output << std::endl;
    std::cout << std::boolalpha << nt::functional::all(func_output == output) << std::noboolalpha << std::endl;
    // std::cout << "should hold "<<output.numel() / (groups * Cout * batch_size) << " of the lines" << std::endl;
    // std::cout << output.view(groups, -1, per_row, Cout).transpose(0,1) << std::endl;


}

void conv_matmult_t(){
    int64_t batch_size = 2;
    int64_t in_channels = 4;
    int64_t out_channels = 5;
    int64_t kernel_sh = 3;
    int64_t x_sh = 20;
    nt::utils::my_tuple stride = 1;
    nt::utils::my_tuple dilation = 1;
    nt::utils::my_tuple padding = 0;

    nt::Tensor x = nt::functional::randn({batch_size, in_channels, x_sh, x_sh});
    nt::Tensor w = nt::functional::randn({out_channels, in_channels, kernel_sh, kernel_sh});
    int64_t Rout = ((x.shape()[-2] + 2 * padding[0] - dilation[0] * (w.shape()[-2] - 1) - 1) / stride[0]) + 1;
	int64_t Cout = ((x.shape()[-1] + 2 * padding[1] - dilation[1] * (w.shape()[-1] - 1) - 1) / stride[1]) + 1;
    nt::Tensor unfolded = nt::functional::unfold(x, {kernel_sh, kernel_sh}, 1, 0, 1, true);
    std::cout <<"unfolded true shape: "<<unfolded.shape() << std::endl;
    nt::Tensor unfolded_2 = nt::functional::unfold(x, {kernel_sh, kernel_sh}, 1, 0, 1, false);
    std::cout <<"unfolded false shape: "<<unfolded_2.shape() << std::endl;
    w = w.view(w.shape()[0], -1);
    std::cout << "w shape: "<<w.shape()<<std::endl;
    nt::Tensor outp_unfold = nt::functional::matmult(unfolded, w, true, true).RowColSwap().view(batch_size, -1, Rout, Cout);
    std::cout << "outp_unfold shape: "<<outp_unfold.shape() << std::endl;
}

void conv_matmult_grouped(){
    int64_t groups = 3;
    int64_t batch_size = 2;
    int64_t in_channels = 4 * groups;
    int64_t out_channels = 5 * groups;
    int64_t kernel_sh = 3;
    int64_t x_sh = 10;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t padding = 0;
    nt::Tensor x = nt::functional::randn({batch_size, in_channels, x_sh});
    nt::Tensor w = nt::functional::randn({out_channels, in_channels/groups, kernel_sh});
    
    nt::Tensor out_functional = nt::functional::conv1d(x, w, stride, padding, dilation, groups);
    std::cout << "out functional shape: "<<out_functional.shape() << std::endl;
    std::cout << "out channels should be: "<<out_channels<<std::endl;

    const int64_t Cout = ((x.shape()[-1] + 2 * padding - dilation * (w.shape()[-1] - 1) - 1) / stride) + 1;

    nt::Tensor inp_unfold = nt::functional::unfold1d(x, w.shape()[-1], dilation, padding, stride, true);
    std::cout << "unfolded"<<std::endl;
    int64_t add = int64_t(x.shape()[1]/groups) * (w.shape()[-1]);
    int64_t k_add = w.shape()[0] / groups;
    nt::Tensor x_parts = inp_unfold.split_axis({nt::range_(0, inp_unfold.shape()[0]), nt::range_(0, add)}).transpose_Tensors(-1,-2).clone();
    nt::Tensor k_parts = w.split_axis({nt::range_(0, k_add)}).view_Tensors(k_add, -1).transpose_Tensors(-1,-2).clone();
    std::cout << x_parts[0].item<nt::Tensor>().shape()<<std::endl;
    std::cout << k_parts[0].item<nt::Tensor>().shape() << std::endl;
    std::cout << "running matrix multiplication"<<std::endl;
    nt::Tensor output = nt::functional::matmult(x_parts, k_parts, false, false).RowColSwap();
    std::cout << "after row col swap shape is "<<output.shape() << std::endl;
    std::cout << "add is: "<<add<<" and k add is "<<k_add<<" and Cout is "<<Cout << std::endl;
    std::cout << "transposing"<<std::endl;
    output = output.view(batch_size, -1, Cout);
    int64_t per_row = output.numel() / (groups * Cout * batch_size);
    output = output.view(groups, -1, per_row, Cout).transpose(0,1).view(x.shape()[0], -1, Cout).contiguous();

    std::cout << "output shape: " << output.shape();
    std::cout << "output: "<<output<<std::endl;
    std::cout << "functional output: "<<out_functional<<std::endl;
    std::cout << std::boolalpha << nt::functional::all(out_functional == output) << std::noboolalpha << std::endl;
}


void conv_gradient_tests(){
    nt::intrusive_ptr<nt::tensor_holder> kernel_hold = nt::make_intrusive<nt::tensor_holder>(nt::Tensor::Null());
    nt::intrusive_ptr<nt::tensor_holder> img_hold = nt::make_intrusive<nt::tensor_holder>(nt::Tensor::Null());

    int64_t groups = 1;
    int64_t batch_size = 2;
    int64_t in_channels = 4 * groups;
    int64_t out_channels = 5 * groups;
    int64_t kernel_sh = 3;
    int64_t x_sh = 10;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t padding = 0;
    nt::Tensor x = nt::functional::randn({batch_size, in_channels, x_sh, x_sh});
    nt::Tensor w = nt::functional::randn({out_channels, in_channels/groups, kernel_sh, kernel_sh});
    
    nt::Tensor out_functional = nt::functional::conv2d(x, w, stride, padding, dilation, groups, img_hold, kernel_hold);

    nt::Tensor dx = nt::functional::zeros_like(x);
    nt::Tensor dw = nt::functional::zeros_like(w);
    nt::Tensor dz = nt::functional::randn(out_functional.shape(), out_functional.dtype()); // gradient
    nt::functional::conv_dimage(dz, kernel_hold->tensor, dx, {kernel_sh, kernel_sh}, {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    nt::functional::conv_dkernel(dz, img_hold->tensor, dw, {x_sh, x_sh}, groups);
    std::cout << "dx is "<<dx<<std::endl;
    std::cout << "dw is "<<dw<<std::endl;
}

void convT_gradient_tests(){
    nt::intrusive_ptr<nt::tensor_holder> kernel_hold = nt::make_intrusive<nt::tensor_holder>(nt::Tensor::Null());
    nt::intrusive_ptr<nt::tensor_holder> img_hold = nt::make_intrusive<nt::tensor_holder>(nt::Tensor::Null());

    int64_t groups = 3;
    int64_t batch_size = 2;
    int64_t in_channels = 4 * groups;
    int64_t out_channels = 5 * groups;
    int64_t kernel_sh = 3;
    int64_t x_sh = 10;
    int64_t stride = 2;
    int64_t dilation = 1;
    int64_t padding = 1;
    int64_t output_padding=1;
    nt::Tensor x = nt::functional::randn({batch_size, in_channels, x_sh, x_sh});
    nt::Tensor w = nt::functional::randn({in_channels, out_channels / groups, kernel_sh, kernel_sh});
    
    nt::Tensor out_functional = nt::functional::conv_transpose2d(x, w, stride, padding, output_padding, dilation, groups, img_hold, kernel_hold);
    
    std::cout << "got conv transpose"<<std::endl;
    nt::Tensor dx = nt::functional::zeros_like(x);
    nt::Tensor dw = nt::functional::zeros_like(w);
    nt::Tensor dz = nt::functional::randn(out_functional.shape(), out_functional.dtype()); // gradient
    nt::functional::convt_dimage(dz, kernel_hold->tensor, dx, {kernel_sh, kernel_sh}, {stride, stride}, {padding, padding}, {output_padding, output_padding}, {dilation, dilation}, groups);
    nt::functional::convt_dkernel(dz, img_hold->tensor, dw, {padding, padding}, {x_sh, x_sh}, groups);
    std::cout << "dx is "<<dx<<std::endl;
    std::cout << "dw is "<<dw<<std::endl;
}


void linear_test(){
    using namespace nt;

    Tensor x = functional::rand(0, 10, {5, 6, 7, 5}, DType::Float32);
    Tensor w = functional::randn({5, 6});
    Tensor b = functional::randn({6});

    Tensor o1 = functional::matmult(x, w) + b;
    Tensor o2 = functional::linear(x, w, b);
    std::cout 
        << std::boolalpha
        << functional::all(o1 == o2)
        << std::noboolalpha
        << std::endl;
    // std::cout << o1 << std::endl;
    // std::cout << o2 << std::endl;
    // in all, the following comes out to around 1e-9, (they are all off by the same amount, and it is always less than 1e-6)
    // this shows that the linear function does work, and it is a lot faster than a matrix multiplication plus the bias
    std::cout << (o1 - o2).sum().item<float>() / o1.numel() << std::endl;
}


void max_test(){
    nt::Tensor example = nt::functional::rand(0, 10, {5, 5}, nt::DType::Float32);
    auto max_r = example.max();

    std::cout << max_r.indices << std::endl;
    std::cout << example << std::endl;
    std::cout << max_r.values << std::endl;
    std::cout << nt::functional::where(max_r.indices) << std::endl;

}


nt::Tensor sample_gumbel(const nt::Tensor& logits){
    nt::Tensor u = nt::functional::rand(0, 1, logits.shape(), logits.dtype()); //uniform (0,1)
    return -nt::functional::log(-nt::functional::log(u + 1e-10));       // Gumbel(0,1)
}


nt::Tensor gumbel_softmax(const nt::Tensor& logits, float tau, bool hard = false) {
    nt::Tensor gumbel_noise = sample_gumbel(logits);
    nt::Tensor y = ((logits) + gumbel_noise) / tau;
    std::cout << "y1: "<<y<<std::endl;
    y = nt::functional::softmax_stable(y, -1); // apply softmax along last dim
    std::cout << "y2: "<<y<<std::endl;

    if (hard) {
        // Straight-through: make y_hard one-hot
        nt::Tensor y_hard = nt::functional::one_hot(nt::functional::argmax(y, -1), y.shape()[-1]).to(y.dtype());
        // Use straight-through estimator
        return (y_hard - y) + y;
    }
    return y;
}


void softmax_test(){
    //nt::Tensor example({15}, nt::DType::Float32);
    nt::Tensor example = nt::functional::rand(1, 200, {10}, nt::DType::Float32);
    //example << 71.5702,85.3064,-0.426085,-0.0928458,2.35356,1.26398,2.367,1.19607,89.1849,2.29037,1.72298,71.1813,93.6324,33.7881,5.83547;
    std::cout << example << std::endl;
    std::cout << nt::functional::softmax_stable(example) << std::endl;
    std::cout << "[MULTI DIM TEST]"<<std::endl;
    nt::Tensor example_2 = nt::functional::rand(1, 200, {3, 10}, nt::DType::Float32);
    std::cout << example_2<<std::endl;
    std::cout << nt::functional::softmax_stable(example_2, -2) << std::endl;
    std::cout << nt::functional::softmax_stable(example_2, -1) << std::endl;

    nt::Tensor running = nt::functional::rand(-200, 200, {3, 10}, nt::DType::Float32);
    // nt::Tensor logits = nt::functional::tanh(running);
    // std::cout << "logits: "<<logits<<std::endl;
    std::cout << "gumbel softmax: "<<gumbel_softmax(running, 1.0, true) << std::endl;
}


inline nt::Tensor __nt__test__symmetric_bilinear__(const nt::Tensor& input, const nt::Tensor& weight){
    return (0.5) * (::nt::functional::matmult(::nt::functional::matmult(weight, input), weight, false, true)
                    + ::nt::functional::matmult(::nt::functional::matmult(weight, input, true, false), weight));
}

inline nt::Tensor __nt__test__attn__symmetric_bilinear__(const nt::Tensor& input, const nt::Tensor& W1, const nt::Tensor& W2){
    nt::Tensor O1 = nt::functional::matmult(W1, input);
    nt::Tensor O2 = nt::functional::matmult(input, W1, false, true);
    nt::Tensor O4 = nt::functional::matmult(W2, O1);
    nt::Tensor O5 = nt::functional::matmult(O2, W2, false, true);
    return O5 + O4;
}

void symmetric_mult_test(){
    using namespace nt;
    Tensor A = functional::rand(1, 3, {10}, DType::int32).to(DType::Float32);
    std::cout << A << std::endl;
    A = (A.view(10, 1) * A.view(1, 10));
    A.fill_diagonal_(0);
    Tensor W = functional::randn({12, 10}, DType::Float32);
    Tensor W2 = functional::randn({10, 12}, DType::Float32);

    std::cout << A << std::endl;
    std::cout << W << std::endl;
    //this works
    // Tensor O = 0.5 * (
    //     functional::matmult(
    //     functional::matmult(W, W, true, false), 
    //     A) +
    //     functional::matmult(
    //     functional::matmult(A, W, false, true), W));

    //this also works:
    // Tensor O = 0.5 * (
    //     functional::matmult(
    //     functional::matmult(
    //     functional::matmult(
    //     functional::matmult(W, W, true, false), A),
    //     W, false, true), W) );
   
    //also works, but W has to be exact same shape as A
    //Tensor O = __nt__test__symmetric_bilinear__(A, W);
    //this works well, allows multiple weights, keeps everything symmetric, and allows hidden dimension
    Tensor O = __nt__test__attn__symmetric_bilinear__(A, W, W2);
    std::cout << O << std::endl;
    std::cout << std::boolalpha <<
        functional::all(O == O.transpose(-1,-2))
        <<std::noboolalpha << std::endl;
    std::cout << O - O.transpose(-1,-2) << std::endl;


}

