#include "src/Tensor.h"
#include "src/dtype/ArrayVoid.h"
#include "src/dtype/DType.h"
#include "src/dtype/DType_enum.h"
#include "src/dtype/DType_list.h"
#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <ios>
#include <iterator>
#include <numeric>
#include <ostream>
#include <chrono>
#include <functional>
#include <sys/_types/_size_t.h>
#include <type_traits>
#include "src/dtype/Scalar.h"
#include "src/dtype/ranges.h"
#include "src/layers/layers.h"
#include "src/refs/ArrayRef.h"
#include "src/refs/SizeRef.h"
#include "src/types/Types.h"
#include <stdio.h>
#include <string.h>

void test_working(){
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
	t = t.view({60});
	t.print();
	
	std::cout<<"t2:"<<std::endl;
	t2.print();
}

void test_permute(){
	nt::Tensor t = nt::functional::arange(2*3*4*5).view({2,3,4,5});
	std::cout<<"original tensor:"<<std::endl;
	t.print();
	nt::Tensor pt = t.permute({2,1,3,0});
	std::cout<<"permuted:"<<std::endl;
	pt.print();
}

void test_transpose(){
	nt::Tensor t = nt::functional::arange(2*3*4*5).view({2,3,4,5});
	std::cout<<"original tensor:"<<std::endl;
	t.print();
	nt::Tensor pt = t.transpose(2,0);
	std::cout<<"transposed:"<<std::endl;
	pt.print();
}

void test_print_indexes(){
	nt::Tensor t = nt::functional::arange(2*3*4).view({2,3,4});
	const std::vector<uint32_t>& _mults = t.strides();
	for(uint32_t i = 0 ; i < _mults.size(); ++i)
		std::cout<<_mults[i]<<" ";
	std::cout<<std::endl;
	uint32_t index = 0;
	for(uint32_t z = 0; z < 2; ++z){
		for(uint32_t r = 0; r < 3; ++r){
			for(uint32_t c = 0; c < 4; ++c){
					std::cout<<t[z][r][c].item<float>()<<" "<<index<<" "<<z*_mults[1] + r*_mults[2] + c<<std::endl;
				++index;
			}
		}
	}
}


template<typename T>
void print(const std::vector<T>& Vec){
	std::cout<<"{";
	for(uint32_t i = 0; i < Vec.size() - 1; ++i)
		std::cout<<Vec[i]<<",";
	std::cout<<Vec.back()<<"}"<<std::endl;
}

void test_flatten(){
    std::vector<uint32_t> _vals = {3,4,5,5,6};
    int8_t _a = -4;
    int8_t _b = -4;
    _a = _a < 0 ? _vals.size() + _a : _a;
    _b = _b < 0 ? _vals.size() + _b : _b;
    uint8_t begin = _a < _b ? _a : _b;
    uint8_t end = _a < _b ? _b : _a;
    end += 1;
    std::cout<<"begin: "<<(int)begin<<" end: "<<(int)end<<std::endl;
    uint32_t mid_dim = std::accumulate(_vals.begin() + begin, _vals.begin() + end, 1.0, std::multiplies<uint32_t>());
    size_t dims = _vals.size() - (end-begin) + 1;
    std::cout<<"mid dim: "<<mid_dim<<std::endl;
    std::vector<uint32_t> n_vals(dims);
    uint32_t sub = 0;
    for(uint32_t i = 0; i < _vals.size(); ++i){
        if(i == begin){
            n_vals[i] = mid_dim;
            i += (end - begin);
	    sub = (end - begin) - 1;
	    std::cout<<"i is now "<<i<<" sub is "<<sub<<std::endl;
        }
        std::cout<<"i is now "<<i<<" "<<_vals.at(i)<<std::endl;
        n_vals[i - sub] = _vals.at(i);
    }
    print(_vals);
    print(n_vals);
}

void tensor_flatten(){
	nt::Tensor t = nt::functional::arange(2*3*4).view({2,3,4});
	std::cout<<"t original:"<<std::endl;
	t.print();
	std::cout<<"flattening"<<std::endl;
	t = t.flatten(-1, -2);
	std::cout<<"t flattened:"<<std::endl;
	t.print();
}
void value_iterator(){
	nt::Tensor t({2,3,4,5});
	float* data = reinterpret_cast<float*>(t.data_ptr());
	for(uint32_t i = 0; i < 10; ++i){
		*(data) = i;
		++data;
	}
	auto begin = t.val_begin();
	for(uint32_t i = 0; i < 10; ++i){
		std::cout<<*begin<<" ";
		++begin;
	}
}

void dtype_tensor_test(){
	nt::Tensor t({2,3,4,5}, nt::DType::TensorObj);
	uint32_t i =0;
	for(auto it = t.val_begin(); it != t.val_end(); ++it){
		*it = nt::functional::arange(3*4, nt::DType::Long).view({3,4});
		std::cout<<i<<std::endl;
		++i;
	}
	std::cout<<"printing after set"<<std::endl;
	std::cout<<t<<std::endl;
	std::cout<<"printed"<<std::endl;
}

void dtype_tensor_long_test(){
	nt::Tensor t = nt::functional::arange(2*3*4*5, nt::DType::Double).view({2,3,4,5});
	std::cout<<t<<std::endl;
}

void dtype_tensor_tensor_test(){
	nt::Tensor t({2}, nt::DType::TensorObj);
	t[0] = nt::functional::arange(3*4, nt::DType::Long).view({3,4});
	t[1] = nt::functional::arange(3*4, nt::DType::Long).view({3,4});
	std::cout<<t<<std::endl;
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

void row_col_swap_test(){
	nt::Tensor t = nt::functional::arange(3*4*5, nt::DType::Double).view({3,4,5});
	nt::Tensor t2 = nt::functional::arange(3*4*5, nt::DType::Double).view({3,4,5});
	std::cout<<t<<std::endl;
	std::cout<<t.RowColSwap()<<std::endl;
	std::cout<<"regular transpose:"<<std::endl;
	std::cout<<t2.transpose(-1,-2)<<std::endl;

}

void Tensor_Split_Axis(){
	nt::Tensor t = nt::functional::arange(2*3*4*5, nt::DType::Double).view({2,3,4,5});
	nt::Tensor t2 = t.split_axis(-1);
	/* std::cout<<t2<<std::endl; */
	std::iota(t2.val_begin(), t2.val_end(), 0.0);
	std::cout<<"after splitting by cols:"<<std::endl;
	std::cout<<t<<std::endl;
	std::cout<<t2.shape()<<std::endl;
	t2 = t.split_axis(-2);
	std::iota(t2.val_begin(), t2.val_end(), 0.0);
	std::cout<<"after splitting by rows:"<<std::endl;
	std::cout<<t<<std::endl;
	std::cout<<t2.shape()<<std::endl;
}

void tensor_cat(){
	nt::Tensor t = nt::functional::arange(2*3*4*5, nt::DType::Double).view({2,3,4,5});
	nt::Tensor t2 = nt::functional::arange(2*3*4*5, nt::DType::Double).view({2,3,4,5});
	auto begin = t.arr_void().tcbegin<double>();
	auto end = t.arr_void().tcend<double>();
	for(;begin != end; ++begin)
		std::cout<<*begin<<" ";
	std::cout<<std::endl;
	/* nt::Tensor t2 = nt::functional::zeros({2,2,4,5}, nt::DType::Double); */
	/* std::cout<<t2<<std::endl; */
	nt::Tensor x = nt::functional::cat(t, t2, 1);
	std::cout<<x<<std::endl;
	begin = x.arr_void().tcbegin<double>();
	end = x.arr_void().tcend<double>();
	for(;begin != end; ++begin)
		std::cout<<*begin<<" ";
	std::cout<<std::endl;
}

void permute_print_strides(){
	nt::Tensor t = nt::functional::arange(5*4*3*2*6).view({5,4,3,2,6});
	nt::Tensor t2 = nt::functional::arange(3*5*4*3*2*6).view({3,5,4,3,2,6});
	bool check = nt::functional::all(t.permute({1,0,3,2,4}) == t2[0].permute({1,0,3,2,4}));
	std::cout<<std::boolalpha<<check<<std::noboolalpha<<std::endl;
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
	/* auto tb = outp.arr_void().tbegin<float>(); */
	/* nt::tdtype_list<const float>  o = t.arr_void().tcbegin<float>(); */
	/* uint32_t index = 0; */
	/* for(uint32_t i = 0; i < 3; ++i){ */
	/* 	for(uint32_t j = 0; j < 3; ++j){ */
	/* 		nt::tdtype_list<const float> o_cpy1 = o; */
	/* 		std::cout << *o_cpy1<<std::endl;; */
	/* 		for(uint32_t k = 0; k < 2; ++k){ */
	/* 			for(uint32_t  l = 0; l < 2; ++l){ */
	/* 				nt::tdtype_list<const float> o_cpy2 = o_cpy1; */
	/* 				for(uint32_t m = 0; m < 4; ++m){ */
	/* 					for(uint32_t n = 0; n < 4; ++n, ++tb, ++o_cpy2){ */
	/* 						*tb *= *o_cpy2; */
	/* 						++index; */
	/* 					} */
	/* 				} */
	/* 			} */
	/* 		std::cout<<"adding "<<32<<std::endl; */
	/* 		o_cpy1 += (4*4); */
	/* 		} */
	/* 	} */
	/* 	std::cout<<"adding "<<(1*2*1*4*4)<<std::endl; */ 
	/* 	o += (1*2*1*4*4); */
	/* } */
	outp *= t;
	std::cout << outp << std::endl;
}

nt::Tensor vector_to_tensor(std::vector<std::vector<float>> v){
	uint32_t rows = v.size();
	uint32_t cols = v[0].size();
	nt::Tensor output({rows, cols});
	for(uint32_t x = 0; x < rows; ++x){
		for(uint32_t y = 0; y < cols; ++y){
			output[x] = v[x][y];
		}
	}
	return std::move(output);
}


void stack_test(){
	nt::Tensor s = nt::Tensor::FromInitializer<3>({ { {1}, {0} }, { {1}, {1} }, { {0}, {1} }, { {0}, {0} } });
	std::cout<<s<<std::endl;
	nt::Tensor rand = nt::functional::randn({3, 2});
	std::cout << rand <<std::endl;
	nt::Tensor outp = nt::functional::matmult(rand, s);
	std::cout << outp <<std::endl;
	nt::Tensor Bias = nt::functional::randn({3,1});
	std::cout<<"Bias:"<<Bias<<std::endl;;
	/* outp[0] += Bias; */
	outp = 1;
	/* std::cout << outp << std::endl; */
	Bias += outp;
	/* std::cout << outp << std::endl; */
	std::cout<<"Bias:"<<Bias<<std::endl;
	std::cout<<outp<<std::endl;
	nt::Tensor cpy = Bias.contiguous();
	std::cout << "Copy: "<<cpy<<std::endl;

	nt::Tensor Z = nt::Tensor::FromInitializer<3>({ { {1}, {0}, {0} }, { {1}, {1}, {0} }, { {0}, {1}, {0} }, { {0}, {0}, {0} } });
	nt::Tensor a_prev = Z.transpose(-1,-2);
	std::cout << "a_prev: "<<a_prev<<std::endl;

	nt::Tensor dZ = nt::Tensor::FromInitializer<3>({ { {-0.0005} }, { {0.9995} }, { {-0.0005} }, { {0.9995} } });

	nt::Tensor dW = nt::functional::matmult(dZ, a_prev);
	std::cout << "dW: "<<dW<<std::endl;
	dW *= 0.001;
	dW /= 4;
	dW.clip_(-5, 5);
	nt::Tensor Weight = nt::functional::zeros({1, 3});
	std::cout << Weight << std::endl;
	Weight += dW;
	std::cout << "new weight: "<<Weight<<std::endl;
}

void equal_test(){
	nt::Tensor o = nt::functional::zeros({3,2,2});
	std::cout<<o<<std::endl;
	o[0] = 1;
	o[1] = 2;
	o[2] = 3;
	std::cout<<o<<std::endl;

	o = nt::functional::zeros({2,1});
	o[0][0] = 1;
	o[1][0] = 0;
	std::cout<<o<<std::endl;
}


void tensor_obj_test(){
	nt::Tensor ex({20}, nt::DType::TensorObj);
	/* std::cout << "ex use count "<< ex.size_use_count()<<std::endl; */
	std::cout<<"made ex"<<std::endl;
	nt::Tensor first = ex[0];
	std::cout << "Set first"<<std::endl;
	std::cout << first.dtype<<std::endl;
	/* std::cout << "first use count "<< first.size_use_count()<<std::endl; */
	std::cout << "ArrayVoid size: "<<first.arr_void().Size()<<std::endl;
	std::cout << first.shape()<<std::endl;
	first.print();


	first = nt::functional::randn({2,2});
	first.print();
	ex[0].print();
}


void p_this_p_test(){
	/* nt::Tensor Bias = nt::functional::zer(3*3*3).view({3,3,3}); */
	nt::Tensor Bias = nt::functional::randn({3,1,3});
	nt::Tensor X = nt::functional::zeros({4,3,3,3});
	std::cout << "made tensors"<<std::endl;
	//I want to do {3,1} += {4,3,3}
	

	nt::Tensor copy = X.contiguous();
	X += Bias;
	X.print();
	std::cout << copy + Bias << std::endl;

	Bias.print();

	/* std::cout << std::boolalpha<<nt::functional::all(next  == Bias)<<std::noboolalpha<<std::endl; */

	/* nt::Tensor last_ex = nt::functional::zeros({3,1,3}); */
	/* /1* X += last_ex; *1/ */
	/* /1* X.print(); *1/ */
	/* last_ex += X; */
	/* last_ex.print(); */

	/* nt::Tensor a = X + Bias; */
	/* a.print(); */
	/* X.print(); */
	/* X += Bias; */
	/* std::cout<<"added"<<std::endl; */
	/* Bias.print(); */
	/* X.print(); */

}


void mat_mult_test(){
	nt::Tensor a = nt::functional::arange({4,2});
	nt::Tensor b = nt::functional::arange({2,5});
	nt::Tensor c = nt::functional::matmult(a, b);
	c.print();
}


//this is a working outline of the simple linear neural network
class my_simple_nn{
	std::vector<nt::Tensor> weights;
	std::vector<nt::Tensor> biases;
	std::vector<nt::Tensor> a_prevs;
	nt::Scalar lr;
	std::vector<uint32_t> topology;
	nt::Tensor dSigmoid(const nt::Tensor& x){
		return x * (1-x);
	}
	
	void Sigmoid(nt::Tensor& x){
		x *= -1;
		x.exp_();
		x += 1;
		x.inverse_();
	}
	public:
		my_simple_nn(std::vector<uint32_t> _topology, float learning_rate=0.1f)
			:topology(std::move(_topology)),
			a_prevs(_topology.size()),
			lr(learning_rate)
		{
		for(uint32_t i = 1; i < topology.size(); ++i){
			weights.push_back(nt::functional::randn({topology[i], topology[i-1]}));
			biases.push_back(nt::functional::randn({topology[i], 1}));
			a_prevs.push_back(nt::functional::zeros({1,1}));
		}
		a_prevs.push_back(nt::functional::zeros({1,1}));
		for(uint32_t i = 0; i < weights.size(); ++i){
			weights[i] = 0.1;
			biases[i] = 0.1;
		}
		}

		bool forward(const nt::Tensor& x){
			a_prevs[0] = x.contiguous();
			nt::Tensor val = nt::functional::matmult(weights[0], x);
			val += biases[0];
			Sigmoid(val);
			for(uint32_t  i = 1; i < weights.size(); ++i){
				/* std::cout << val<<std::endl; */
				a_prevs[i] = val.contiguous();
				val = nt::functional::matmult(weights[i], val);
				val += biases[i];
				Sigmoid(val);
			}
			/* std::cout << val<<std::endl; */
			a_prevs[weights.size()] = val.contiguous();
			return true;
		}

		bool backward(const nt::Tensor& target){
			nt::Tensor dZ = target - a_prevs[weights.size()];
			std::cout<<"First dZ:"<<dZ<<std::endl;
			/* std::cout << "sum: "<<dZ.sum().item<float>()<<std::endl; */
			/* std::cout<<a_prevs[weights.size()]<<std::endl; */
			for(int i = weights.size() -1; i >= 0; i--){
				nt::Tensor trans = weights[i].transpose(-1,-2);
				nt::Tensor prevErrors = nt::functional::matmult(trans, dZ);
				std::cout << "Prev Errors "<<i<<": "<<prevErrors<<std::endl;
				nt::Tensor dOutput = dSigmoid(a_prevs[i+1]);
				nt::Tensor grads = dZ * dOutput;
				grads *= lr;
				nt::Tensor dW = nt::functional::matmult(grads, a_prevs[i].transpose(-1,-2));
				grads.clip_(-5,5);
				std::cout << "grads: "<<grads<<std::endl;
				dW.clip_(-5,5);
				std::cout << "dW: "<<dW<<std::endl;
				weights[i] += dW;
				biases[i] += grads;
				dZ = prevErrors;
			}
			return true;
		}

		nt::Tensor eval(const nt::Tensor& x){
			nt::Tensor val = nt::functional::matmult(weights[0], x);
			val += biases[0];
			Sigmoid(val);
			for(uint32_t  i = 1; i < weights.size(); ++i){
				val = nt::functional::matmult(weights[i], val);
				val += biases[i];
				Sigmoid(val);	
			}
			return val;
		}
	

};


void transpose_test(){
	nt::Tensor a = nt::functional::arange({4,4,4});
	nt::Tensor b = a.permute({2,1,0});
	b.print();
	b = b[1];
	b.print();
	b.permute({1,0}).print();
	a.permute({2,1,0})[1].permute({1,0})[3].print();
}

void pseudo_nn_simple_a(){
	nt::Tensor inp_stacked = nt::Tensor::FromInitializer<3>({ { {1}, {0} }, { {1}, {1} }, { {0}, {1} }, { {0}, {0} } });
	nt::Tensor outp_stacked = nt::Tensor::FromInitializer<3>({ { {0} }, { {1} }, { {0} }, { {1} } });
	my_simple_nn nn({2, 3, 1});
	uint32_t epoch = 10000;
	for(uint32_t i = 0; i < epoch; ++i){
		nn.forward(inp_stacked);
		nn.backward(outp_stacked);
	}
	for(uint32_t i = 0; i < 4; ++i){
		std::cout << nn.eval(inp_stacked[i]) << std::endl;
	}
}

//this also works using the layers module
void pseudo_nn_simple(){
	
	nt::Tensor inp_stacked = nt::Tensor::FromInitializer<3>({ { {1}, {0} }, { {1}, {1} }, { {0}, {1} }, { {0}, {0} } });
	nt::Tensor outp_stacked = nt::Tensor::FromInitializer<3>({ { {0} }, { {1} }, { {0} }, { {1} } });
	
	nt::layers::Sequential nn(nt::layers::Linear(2,3,1), 
			nt::layers::Sigmoid(),
			nt::layers::Linear(3,1,1),
			nt::layers::Sigmoid());
	uint32_t epoch = 10000;
	for(uint32_t q = 0; q < epoch; ++q){
		nt::Tensor output = nn.forward(inp_stacked);
		nn.backward(nt::layers::loss_error(output, outp_stacked));
	}
	nn.eval(inp_stacked).print();
}


void relu_example(){
	nt::Tensor x = nt::functional::randint(-10, 10, {4,4,4}, nt::DType::Float);
	x.print();
	x[x < 0] = 0;
	x.print();
}

void pad_test(){
	nt::Tensor x = nt::functional::arange({4,4,4,4});
	nt::Tensor p = x.pad({1,1,1,1});
	p.print();
}

void flip_test(){
	nt::Tensor x = nt::functional::arange({4,5,4});
	x.print();
	x.flip_(-2).print();
	x.flip_(-1).print();
	nt::Tensor flipped = x.flip_(-1).flip_(-2);
	flipped.print();
}

void dilate_test(){
	nt::Tensor x = nt::functional::arange({4,5,4}) + 1;
	nt::Tensor dil = x.dilate_(2);
	dil.print();
	x.print();
	x *= 2.5;
	dil.print();
	dil += 1.5;
	x.print();
	dil.arr_void().fill_ptr_(0);
	dil.print();
}

void max_sum_test(){
	nt::Tensor summing({3,4,4,4}, nt::DType::Float);
	summing = 1;
	nt::Tensor summed = summing.sum(-2);
	summed.print();

	nt::Tensor maxing = nt::functional::randint(-5, 5, {3,4,4,4}, nt::DType::Float);
	maxing.print();
	nt::Tensor maxed = maxing.max(-2);
	maxed.print();
}

void conv_test(){
	const nt::Tensor rn = nt::functional::randn({4,4,300,300});
	std::cout<<"made rn"<<std::endl;
	const nt::Tensor r2 = rn.split_axis_1_();
	std::cout<<"split r2"<<std::endl;
	/* rn.print(); */
	/* nt::Tensor parts = rn.split_axis(-3); */
	/* nt::Tensor* begin = reinterpret_cast<nt::Tensor*>(parts.data_ptr()); */
	/* nt::Tensor* end = begin + parts.numel(); */
	/* for(;begin != end; ++begin){ */
	/* 	begin->print(); */
	/* } */
	const nt::Tensor mi = nt::functional::arange({3,3,4});
	mi.print();
	const nt::Tensor m2 = mi.split_axis_1_();
	const nt::Tensor* begin = reinterpret_cast<const nt::Tensor*>(m2.data_ptr());
	const nt::Tensor* end = begin + m2.numel();
	for(;begin != end; ++begin)
		begin->print();
	/* nt::Tensor img({1,3,300,300}); */
	/* img = 1; */
	/* nt::Tensor kernel = nt::functional::randn({20,3,3,3}); */
	/* nt::Tensor outp = nt::functional::conv2d(img, kernel); */
	/* std::cout << outp.shape() << std::endl; */
}

/* #include "src/functional/functional_matmult.h" */
/* #include "src/dtype/ArrayVoid.hpp" */

void better_matMult(){
	std::cout << "getting tensors"<<std::endl;
	const nt::Tensor a = nt::functional::arange({2,3,4}, nt::DType::Float32);
	const nt::Tensor b = nt::functional::arange({8,2,4,5}, nt::DType::Float32);
	nt::Tensor output = nt::functional::matmult(a, b);
	output.print();
}

void unfold_test(){
	nt::Tensor a = nt::functional::arange({3,3,4});
	while(true){
		std::string inp;
		std::cout << "doing dim: ";
		std::cin >> inp;
		if(inp == "q")
			break;
		if(inp[0] == 's'){
			std::cout<<"changing size"<<std::endl;
			std::string delimeter = ",";
			inp.erase(0,1);
			size_t pos = 0;
			std::string token;
			std::vector<uint32_t> n_size;
			while((pos = inp.find(delimeter)) != std::string::npos){
				token = inp.substr(0, pos);
				n_size.push_back(std::stoi(token));
				inp.erase(0, pos+delimeter.length());
			}
			n_size.push_back(std::stoi(inp));
			a = nt::functional::arange(nt::SizeRef(std::move(n_size)));
			continue;
		}
		bool neg = false;
		if(inp[0] == '-'){
			neg = true;
			inp.erase(0, 1);
		}
		if(inp.length() == 0){
			std::cout<<"length error"<<std::endl;
			break;
		}
		while((inp[0] < '0' || inp[0] > '1') && inp.length() > 0 && !neg){
			std::cout<<"begin erase"<<std::endl;
			inp.erase(0,1);
		}
		int val = std::stoi(inp);
		if(neg)
			val *= -1;
		nt::Tensor unfolded = a.unfold(val, 2, 3);
		unfolded.print();
	}
	/* std::cout << "doing dim: -1"<<std::endl; */
	/* a.unfold(-1,2,1); */
}


void unfold_layer_test(nt::Tensor a){
	/* nt::Tensor a = nt::functional::arange({2,2,6,6}); */
	auto layer = nt::layers::Unfold(3, 1, 0, 1, false);
	std::cout<<"unfolding"<<std::endl;
	nt::Tensor x = layer.forward(a);
}


int current_test(std::function<void()> test_func){
	std::cout<<"starting test"<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	try{
		test_func();
	}
	catch (const std::exception &exc){
		std::cerr << exc.what();
		return 1;
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout<<"finished execution of function successfully in "<<duration.count()<<std::endl;
	return 0;
}

int current_test(std::function<void(nt::Tensor)> test_func, nt::Tensor x){
	std::cout<<"starting test"<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	try{
		test_func(std::move(x));
	}
	catch (const std::exception &exc){
		std::cerr << exc.what();
		return 1;
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout<<"finished execution of function successfully in "<<duration.count()<<std::endl;
	return 0;
}


