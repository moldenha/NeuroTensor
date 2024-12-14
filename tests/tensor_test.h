//this is a file for testing the tensor class and some of it's functionality
//needs to be extended
#include "../src/Tensor.h"

#include <iostream>
#include <string>


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



