#include "../src/Tensor.h"
#include "../src/functional/functional.h"
#include "../src/layers/TensorGrad.h"
#include "../src/layers/layers.h"
#include <memory>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include <variant>
#include <unordered_map>
#include "../src/dtype/compatible/DType_compatible.h"
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <optional>


int fold_autograd_test(){
	nt::TensorGrad A(nt::functional::randn({10,3,4,2}));
	nt::TensorGrad W(nt::functional::randn({3,2,4}));

	nt::TensorGrad B = nt::functional::matmult(A[nt::my_range(2,4)], W);
	nt::TensorGrad C = nt::functional::matmult(A[nt::my_range(6,8)], W);

	std::cout << B.shape() << ',' << C.shape();
	nt::TensorGrad fB_1 = B[0];
	nt::TensorGrad fB_2 = B[1];
	nt::TensorGrad fC_1 = C[0];
	nt::TensorGrad fC_2 = C[1];

	nt::TensorGrad Combo = (fB_1 + fB_2 + fC_1 + fC_2).unsqueeze(0);

	nt::TensorGrad folded = nt::functional::unfold(Combo, 2, 1, 0, 1);
	std::cout << "folded shape: "<<folded.shape()<<std::endl;

	nt::Tensor dF = nt::functional::randn(folded.shape());
	folded.zero_grad();
	folded.backward(dF);
	std::cout << "Gradient of A: "<<A.grad_value()<<std::endl;
	std::cout << "Gradient of W: "<<W.grad_value() << std::endl;
	std::cout << "Gradient of fb_1: "<<fB_1.grad_value() << std::endl;
	std::cout << "Gradient of fc_1: "<<fC_1.grad_value() << std::endl;
	std::cout << "Gradient of B: "<<B.grad_value()<<std::endl;
	std::cout << "Gradient of C: "<<C.grad_value()<<std::endl;

	return 0;
}


//this is basically to test branching functions
int relu_autograd_test(){

	

	nt::TensorGrad A(nt::functional::randn({3,4,2}));
	nt::TensorGrad myScalar(0.5f);
	A.tensor[A.tensor < 0.01] *= -1;
	nt::Tensor track = nt::functional::where(A <= 0);
	auto shape = A[A <= 0].shape();
	nt::TensorGrad Add_Branch(nt::functional::randn(shape));
	nt::TensorGrad Mult_Branch(nt::functional::randn(shape));
	std::cout << "A: "<<A<<std::endl;
	nt::TensorGrad W(nt::functional::randn({3,2,3}));

	A[track] = 0;
	std::cout << "A post relu: "<<A<<std::endl;
	std::cout << "adding "<<A[track].shape() << " and "<< Add_Branch.shape() << std::endl;
	A[track] += Add_Branch;
	std::cout << "multiplying "<<A[track].shape() << " and "<< Mult_Branch.shape() << std::endl;
	A[track] *= Mult_Branch;
	std::cout << "adding again..."<<std::endl;
	std::cout << "adding "<< A[track].shape() << " and " << shape << std::endl;
	A[track] += (nt::functional::randn(shape) * myScalar) * (nt::functional::randn(shape) + myScalar);
	std::cout << "now getting out..."<<std::endl;
	nt::TensorGrad out = nt::functional::matmult(A, W);
	std::cout << "out parent size is "<<out.parents.size()<<std::endl;
	nt::Tensor dt = nt::functional::randn(out.shape());
	std::cout << "A children size: "<<A.children->size()<<std::endl;
	out.zero_grad();
	std::cout << "zero gradded"<<std::endl;
	out.backward(dt);
	std::cout << "Gradient of A: "<<A.grad_value()<<std::endl;
	std::cout << "A children size: "<<A.children->size()<<std::endl;
	std::cout << "out parents size: "<<out.parents.size()<<std::endl;
	std::cout << out.parents[0]->shape()<<std::endl;
	std::cout << out.parents[1]->shape()<<std::endl;
	std::cout << out.parents[0]->grad_value()<<std::endl;
	std::cout << out.parents[0]->children->size()<<std::endl;

	std::cout << "Add branch grad: "<<Add_Branch.grad_value()<<std::endl;
	std::cout << "Mult branch grad: "<<Mult_Branch.grad_value() << std::endl;
	std::cout << "myScalar grad: "<<myScalar.grad_value() << std::endl;
	return 0;
}


int view_autograd_test(){
	//there is something weird going on with this, look into the mkl matmult file
	/* nt::Tensor a = nt::functional::randn({3,2,7}); */
	/* nt::Tensor b = nt::functional::randn({3,2,6}); */
	/* std::cout << nt::functional::matmult(b, a,1,0); */
	/* nt::Tensor c = nt::functional::randn({2,7}); */
	/* nt::Tensor d = nt::functional::randn({2,6}); */
	/* std::cout << nt::functional::matmult(d, c,1,0); */
	/* std::cout << a << std::endl; */
	/* std::cout << a[0] << std::endl; */
	/* std::cout << a[0].numel()<<std::endl; */
	nt::TensorGrad A(nt::functional::randn({3,4,5}));
	
	nt::layers::Linear p1_a(5, 600);
	nt::layers::Linear p1_b(600, 4);

	nt::layers::Linear p2_a(10, 1000);
	nt::layers::Linear p2_b(1000, 8);
	
	nt::TensorGrad P1_A = p1_a.forward(A[0]);
	nt::TensorGrad P1_B = p1_b.forward(P1_A);

	nt::TensorGrad P2_A = p2_a.forward(A[1].view(2,10));
	nt::TensorGrad P2_B = p2_b.forward(P2_A);
	nt::TensorGrad P2_V = P2_B.view(P1_B.shape());
	nt::TensorGrad X = P2_V + P1_B;


	/* nt::TensorGrad X = p1_b.forward(p1_a.forward(A[0])); */
	/* X += p2_b.forward(p2_a.forward(A[1].view(2,10))).view(X.shape()); */
	
	nt::Tensor dX = nt::functional::randn(X.shape());
	X.zero_grad();
	X.backward(dX);

	
	/* std::cout << "Gradient of X: "<<X.grad_value()<<std::endl; */
	/* std::cout << "Gradient of P2_V: "<<P2_V.grad_value() << std::endl; */
	/* std::cout << "Gradient of P2_B: "<<P2_B.grad_value() << std::endl; */
	/* std::cout << "Gradient of P2_A: "<<P2_A.grad_value() << std::endl; */
	/* std::cout << "Gradient of P1_B: "<<P1_B.grad_value() << std::endl; */
	/* std::cout << "Gradient of P1_A: "<<P1_A.grad_value() << std::endl; */
	std::cout << "Gradient of A: "<<A.grad_value()<<std::endl;
	



	/* std::cout << "path1 A:";p1_a.print_grads(); */
	/* std::cout << "path1 B:";p1_b.print_grads(); */
	/* std::cout << "path2 A:";p2_a.print_grads(); */
	/* std::cout << "path2 B:";p2_b.print_grads(); */
	return 0;

}


//this works where all the gradients are updated
//obviously, there is a huge exploding gradient just because of the design
//but, it does produce gradients for all of the layers involved and A,B,C,D,E and X
//so this is a successful test which is nice
int this_operation_autograd_test(){
	
	//this is going to be a multi-path thing
	nt::layers::Linear p1_a(5, 6);
	nt::layers::Linear p1_b(6, 4);
	
	nt::layers::Linear p2_a(5,2);
	nt::layers::Linear p2_b(2,4);

	nt::layers::Linear p3_a(5, 7);
	nt::layers::Linear p3_b(7, 4);
	
	nt::layers::Linear p4_a(5, 7);
	nt::layers::Linear p4_b(7, 4);
	
	nt::layers::Linear p5_a(5, 60);
	nt::layers::Linear p5_b(60, 4);
	
	nt::TensorGrad A(nt::functional::randn({3,3,5}));
	nt::TensorGrad B(nt::functional::randn({3,3,4}));
	nt::TensorGrad C(nt::functional::randn({3,3,4}));
	nt::TensorGrad D(nt::functional::randn({3,3,4}));
	nt::TensorGrad E(nt::functional::randn({3,3,4}));

	nt::TensorGrad X = p1_b.forward(p1_a.forward(A));
	X += (p2_b.forward(p2_a.forward(A)) / B);
	X -= (p3_b.forward(p3_a.forward(A)) - C);
	X /= (p4_b.forward(p4_a.forward(A)) * D);
	X *= (p5_b.forward(p5_a.forward(A)) + E);

	X.zero_grad();

	nt::Tensor dX = nt::functional::randn({3,3,4});
	X.backward(dX);
	
	std::cout << "dX: "<<dX<<std::endl;
	std::cout << "Gradient of X: "<<X.grad_value()<<std::endl;
	std::cout << "Gradient of A: "<<A.grad_value()<<std::endl;

	/* std::cout << "path1 A:";p1_a.print_grads(); */
	/* std::cout << "path1 B:";p1_b.print_grads(); */
	/* std::cout << "path2 A:";p2_a.print_grads(); */
	/* std::cout << "path2 B:";p2_b.print_grads(); */
	/* std::cout << "path3 A:";p3_a.print_grads(); */
	/* std::cout << "path3 B:";p3_b.print_grads(); */

	std::cout << "Gradient of B: "<<B.grad_value()<<std::endl;
	std::cout << "Gradient of C: "<<C.grad_value()<<std::endl;
	std::cout << "Gradient of D: "<<D.grad_value()<<std::endl;
	std::cout << "Gradient of E: "<<E.grad_value()<<std::endl;

	return 0;
}



int linear_autograd_test(){
	int64_t in_features = 5;
	nt::layers::Linear a(in_features, 6);
	nt::layers::Linear b(6, 7);
	nt::layers::Linear c(7, 5);
	nt::TensorGrad A(nt::functional::randn({3,3,in_features}));

	A = a.forward(A);
	A = b.forward(A);
	A = c.forward(A);
	A.zero_grad();

	nt::Tensor dA = nt::functional::randn(A.tensor.shape());
	std::cout << "dA: "<<dA<<std::endl;
	A.backward(dA);


	std::cout << "Gradient of A: "<<A.grad_value()<<std::endl;
	/* std::cout << "Linear a: ";a.print_grads(); */
	/* std::cout << "Linear b: ";b.print_grads(); */
	/* std::cout << "Linear c: ";c.print_grads(); */
	return 0;
}


int test_autograd_original(){
    nt::TensorGrad a(nt::functional::randn({2,3,4}));
    nt::TensorGrad b(nt::functional::randn({2,3,4}));
    nt::TensorGrad c(nt::functional::randn({2,4,2}));

    nt::TensorGrad d = a + b;
    nt::TensorGrad e = d * b;
    nt::TensorGrad f = nt::functional::matmult(e, c);

    f.zero_grad();

    nt::Tensor dx = nt::functional::randn({2,3,2});
    std::cout <<"dx: " << dx << std::endl;

    f.backward(dx);
    
    std::cout << "finished doing backward pass, now evaluating gradients:"<<std::endl;
    std::cout << "Gradient of a: " << a.grad_value() << std::endl;
    std::cout << "Gradient of b: " << b.grad_value() << std::endl;
    std::cout << "Gradient of c: " << c.grad_value() << std::endl;
    std::cout << "Gradient of d: " << d.grad_value() << std::endl;
    std::cout << "Gradient of e: " << e.grad_value() << std::endl;
    std::cout << "Gradient of f: " << f.grad_value() << std::endl;

    return 0;
}


int test_autograd(){
	relu_autograd_test();
	return 0;
}


