#include <nt/Tensor.h>
#include <nt/functional/functional.h>
#include <nt/nn/TensorGrad.h>
#include <nt/nn/layers.h>
#include <nt/nn/Loss.h>
#include <memory>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include <variant>
#include <unordered_map>
#include <nt/dtype/compatible/DType_compatible.h>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <optional>


int fold_autograd_test(){
	nt::TensorGrad A(nt::functional::randn({10,3,4,2}));
	nt::TensorGrad W(nt::functional::randn({3,2,4}));

	nt::TensorGrad B = nt::functional::matmult(A[nt::range_(2,4)], W);
	nt::TensorGrad C = nt::functional::matmult(A[nt::range_(6,8)], W);

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
	A.detach()[A.detach() < 0.01] *= -1;
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
	std::cout << "out parent size is "<<out.parents->size()<<std::endl;
	nt::Tensor dt = nt::functional::randint(0, 1, out.shape(), nt::DType::Float32); // this acts as the error
	/* dt[dt < 0.5] = 0; */
	/* dt[dt != 0] = 1; */
	std::cout << "dt: "<<dt<<std::endl;
	std::cout << "A children size: "<<A.children->size()<<std::endl;
	out.zero_grad();
	std::cout << "zero gradded"<<std::endl;
	out.backward(dt);
	std::cout << "Gradient of A: "<<A.grad_value()<<std::endl;
	std::cout << "A children size: "<<A.children->size()<<std::endl;
	std::cout << "out parents size: "<<out.parents->size()<<std::endl;
	// std::cout << out.parents[0]->shape()<<std::endl;
	// std::cout << out.parents[1]->shape()<<std::endl;
	// std::cout << out.parents[0]->grad_value()<<std::endl;
	// std::cout << out.parents[0]->children->size()<<std::endl;

	std::cout << "Add branch grad: "<<Add_Branch.grad_value()<<std::endl;
	std::cout << "Mult branch grad: "<<Mult_Branch.grad_value() << std::endl;
	std::cout << "myScalar grad: "<<myScalar.grad_value() << std::endl;

	std::cout << "Add branch before: "<<Add_Branch<<std::endl;
	Add_Branch.update();
	std::cout << "Add branch after: "<<Add_Branch<<std::endl;
	std::cout << "Mult branch before: "<<Mult_Branch<<std::endl;
	Mult_Branch.update();
	std::cout << "Mult after before: "<<Mult_Branch<<std::endl;
	std::cout << "myScalar before: "<<myScalar<<std::endl;
	myScalar.update();
	std::cout << "myScalar after: "<<myScalar<<std::endl;
	std::cout << "A before: "<<A<<std::endl;
	A.update();
	std::cout << "A After: "<<A<<std::endl;



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

	nt::Tensor dA = nt::functional::randn(A.detach().shape());
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


int test_autograd_cat(){
	nt::TensorGrad a(nt::functional::randn({2, 3, 4}));
	nt::TensorGrad b(nt::functional::randn({3, 3, 4}));
	std::cout << "a: "<<a<<std::endl;
	nt::TensorGrad a_split = a.split_axis(0);
	std::cout << "a_split: "<<a_split<<std::endl;
	std::cout << "a_split grad: "<<a_split.grad->tensor<<std::endl;
	b.zero_grad();
	std::vector<nt::TensorGrad> vec({a, b});
	nt::TensorGrad catted = nt::functional::cat(std::move(vec));
	std::cout << catted << std::endl;
	catted.backward(nt::functional::randn({5,3,4}));
	std::cout << "a grad: "<<a.grad->tensor << std::endl;
	std::cout << "a_split grad: "<<a_split.grad->tensor<<std::endl;
	std::cout << "b grad: "<<b.grad->tensor << std::endl;

	std::cout << std::endl << std::endl << "a_split[1]: "<<a_split[1] << std::endl;
	std::cout << " a_split[1].grad: "<<a_split[1].grad->tensor << std::endl;
	return 0;

}


//this function can take 2 activation functions and show that they are able to reduce the loss
//this is not meant to be a solid model
//there are a few different things implemented to help with exploding gradients:
//  - weights are pre-normalized and multiplied by 0.01
//  - gradients are clipped
//  - once the loss starts to increase again after enough iterations, the training stops
//these are implemented because this function is only meant to make sure that the activation
//functions work, properly reduce the loss, and that the autograd properly handles gradients
//it is not meant to be a good model or anything like that
template<typename ActivationFunction1, typename ActivationFunction2>
bool activation_function_test(ActivationFunction1&& func1, ActivationFunction2&& func2, 
                              nt::DType dt = nt::DType::Float32,
                              nt::Tensor wanted = nt::functional::rand(0, 1, {3, 300}, nt::DType::Float32),
                              int iterations = 500, float lr = 0.001, int it_greater=10,
                              bool clip = true, bool verbose_grad = false,
                              nt::Scalar clip_min=-0.1, nt::Scalar clip_max=0.1,
                              std::function<nt::Tensor(nt::SizeRef, nt::DType)> init = nt::functional::randn,
                              int64_t hidden = 5){
    using namespace nt;
    auto critereon = loss::MSE;
    const int64_t& out_cols = wanted.shape()[-1];
    //the reason they are multiplied by 0.01 is because 
    //(especitally if out cols is small) the weights are skewed
    //this makes functions like softmax sometimes not be able to properly reduce the loss
    TensorGrad weight1(init({10, hidden}, dt));
    TensorGrad bias1(init({hidden}, dt));
    TensorGrad weight2(init({hidden, out_cols}, dt));
    TensorGrad bias2(init({out_cols}, dt));
    float loss1, lossl, curl = 1.0;
    //sometimes values were wanted to be printed
    auto val_grad_getter = [](const TensorGrad& tg)->float{
        return std::abs(tg.grad->tensor).max().values.toScalar().to<float>();
    };
    auto update = [&lr, &weight1, &bias1, &weight2, &bias2, &clip, &val_grad_getter, &verbose_grad, &curl,
                    &clip_min, &clip_max](){
        //clipping helps with gradients exploding
        if(verbose_grad){
            std::cout << "("
                << val_grad_getter(weight1) << ',' 
                << val_grad_getter(weight2) << ',' 
                << val_grad_getter(bias1) << ',' 
                << val_grad_getter(bias2) << ')' << std::endl;

        }
        if(clip){
            weight1.grad->tensor.clip_(clip_min,clip_max) *= lr;
            bias1.grad->tensor.clip_(clip_min, clip_max) *= lr;
            weight2.grad->tensor.clip_(clip_min, clip_max) *= lr;
            bias2.grad->tensor.clip_(clip_min, clip_max) *= lr;
        }else{
            // if(val_grad_getter(weight1) < 1e-3 && curl > 0.4){
            //     weight1.grad->tensor *= 10.0;
            //     bias1.grad->tensor *= 10.0;
            //     weight2.grad->tensor *= 10.0;
            //     bias2.grad->tensor *= 10.0;
            // }else{
                weight1.grad->tensor *= lr;
                bias1.grad->tensor *= lr;
                weight2.grad->tensor *= lr;
                bias2.grad->tensor *= lr;
            // }
        }
        weight1.update();
        bias1.update();
        weight2.update();
        bias2.update();
    };

    auto run = [&weight1, &bias1, &weight2, &bias2, &func1, &func2](const TensorGrad& x){
        TensorGrad x1 = functional::linear(x, weight1, bias1);
        // TensorGrad x1 = functional::matmult(x, weight1) + bias1;
        TensorGrad f_x1 = func1(x1);
        TensorGrad x2 = functional::linear(f_x1, weight2, bias2);
        // TensorGrad x2 = functional::matmult(f_x1, weight2) + bias2;
        TensorGrad f_x2 = func2(x2);
        return std::move(f_x2);
    };

    TensorGrad original(functional::rand(0, 20, {3, 10}, dt));
    int i;
    for(i = 0; i < iterations ; ++i){
        TensorGrad out = run(original);
        TensorGrad loss = critereon(out, wanted);
        //eventually there is an exploding gradient
        //the point of this is purely to see if the activation function works and the
        //autograd handles the gradients properly
        //not to make a working model
        //if(loss.item().to<float>() == 0.0){lossl = 0.0; break;}
        if(it_greater > 0 && i > it_greater && loss.item().to<float>() > curl){
            lossl = curl; break;
        }
        curl = loss.item().to<float>();
        std::cout << "loss["<<i<<"]:\t"<<loss.item();
        if(!verbose_grad)
            std::cout << std::endl;
        if(i == 0){
            loss1 = curl;
            std::cout << out << std::endl;
        }
        if(i == iterations-1){
            lossl = loss.item().to<float>();
        }
        if(curl < 0.1) {lr = 0.001;}
        // if(loss.item().to<float>() == 0.0){continue;}
        loss.backward();
        update();
    }
    std::cout << run(original) << std::endl;
    std::cout << wanted << std::endl;
    std::cout << original << std::endl;
    std::cout << weight1 << std::endl;
    std::cout << iterations << std::endl;
    std::cout << i << std::endl;
    std::cout << loss1 << std::endl;
    std::cout << lossl << std::endl;
    return std::abs(lossl) < std::abs(loss1);
}


//works properly
bool test_softmax_activation(){
    auto func1 = [](const nt::TensorGrad& x){return x;}; 
    auto func2 = [](const nt::TensorGrad& x){return nt::functional::softmax(x, -1);}; 
    nt::DType dt = nt::DType::Float32;
    nt::Tensor wanted = nt::functional::zeros({3, 3}, nt::DType::Float32);
    wanted[1][0] = 1;
    wanted[0][2] = 1;
    wanted[2][1] = 1;
    return activation_function_test(func1, func2, dt, wanted, 2000, 0.01, 10, true, false,
                                    -.1, .1, [](nt::SizeRef sz, nt::DType dt){return nt::functional::randn(sz, dt) * 0.01;});
}

bool test_gumbel_softmax_activation(){
    auto func1 = [](const nt::TensorGrad& x){return x;}; 
    auto func2 = [](const nt::TensorGrad& x){
        // return nt::functional::softmax(x, -1);
        // auto nx = nt::functional::tanh(x) * 10;
        return nt::functional::gumbel_softmax(x, 1.0, true);
    }; 
    nt::DType dt = nt::DType::Float64;
    nt::Tensor wanted = nt::functional::zeros({3, 3}, dt);
    wanted[1][0] = 1;
    wanted[0][2] = 1;
    wanted[2][1] = 1;
    return activation_function_test(func1, func2, dt, wanted, 1000, 0.01, -1, false, false, -10.0, 10.0,
                                    [](nt::SizeRef sz, nt::DType dt){return nt::functional::rand(0, 0.3, sz, dt);}, 7);

}

int test_autograd(){
	relu_autograd_test();
	return 0;
}





