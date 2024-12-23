![github_logo](https://github.com/user-attachments/assets/19861561-d196-4b37-b849-edf2b9a290a6)
<hr>

Version: v0.1.0

NeuroTensor is a tensor framework that has a tda and autograd wrapper built in. The syntax follows PyTorch's closely, but in C++. Current version is 0.1.0 and there is still much to come for added support and testing. This is an early beta version, with an end goal being a framework for computational neuroscience tools.


## Tensor Class

The Tensor class is seperate from the autograd. It is a standalone wrapper, that uses the `nt::DType` enum to switch between different types, and has a wide range of function support. A detailed documentation is to come.

This is an example function pulled from the tests file, showing how it is able to work:

```C++
//some basic syntax and functionality
nt::Tensor t = nt::functional::randn({3,4,5}); //creates a 3x4x5 float tensor
t[1] += 1;
t[t < 0.01] *= -1;
nt::Tensor t2 = t[2];
t2[t2 > 0].exp_(); //also modifies t
t2 = t2.to(nt::DType::Double);
std::cout << t2 << std::endl;

```

## The Autograd

The autograd is a dynamic wrapper class called `TensorGrad`. The `TensorGrad` is being adapted to have all the same functionality as the `Tensor` class, but also tracking gradients. This is an example from testing to make sure that branching works.

```C++
nt::TensorGrad A(nt::functional::randn({3,4,2}));
nt::TensorGrad myScalar(0.5f);
//this operation is done in order to not track the gradient of this operation
A.tensor[A.tensor < 0.01] *= -1;
nt::Tensor track = nt::functional::where(A <= 0);
auto shape = A[A <= 0].shape();
nt::TensorGrad Add_Branch(nt::functional::randn(shape));
nt::TensorGrad Mult_Branch(nt::functional::randn(shape));
std::cout << "A: "<<A<<std::endl;
nt::TensorGrad W(nt::functional::randn({3,2,3}));

A[track] = 0;
std::cout << "A post relu: "<<A<<std::endl;
A[track] += Add_Branch;
A[track] *= Mult_Branch;
A[track] += (nt::functional::randn(shape) * myScalar) * (nt::functional::randn(shape) + myScalar);
nt::TensorGrad out = nt::functional::matmult(A, W);
std::cout << "out parent size is "<<out.parents.size()<<std::endl;
nt::Tensor dt = nt::functional::randn(out.shape());
std::cout << "A children size: "<<A.children->size()<<std::endl;
out.zero_grad();
out.backward(dt);
std::cout << "Gradient of A: "<<A.grad_value()<<std::endl;
std::cout << "Add branch grad: "<<Add_Branch.grad_value()<<std::endl;
std::cout << "Mult branch grad: "<<Mult_Branch.grad_value() << std::endl;
std::cout << "myScalar grad: "<<myScalar.grad_value() << std::endl;

Add_Branch.update();
Mult_Branch.update();
myScalar.update();
A.update();
std::cout << "A is now: "<< A << std::endl;
std::cout << "Add_Branch is now: "<< Add_Branch << std::endl;
std::cout << "Mult_branch is now: "<<Mult_Branch << std::endl;
std::cout << "myScalar is now: "<<myScalar << std::endl;
```

## Layer Class

There is an `nt::Layer` class with a built-in reflection wrapper. Each custom layer or model must be a class that inherits from `nt::Module`. Gradients are automatically tracked, but can be manually created as well by overloading the `backward` function. The `eval` function can also be overloaded if there are differences in the way that the layer or model behaves based on if it is in `eval` mode beyond just not tracking the gradient. Below is an example making of a custom layer:

```C++
//creation of a custom layer
class Example : public nt::Module{
	public:
		nt::Layer l1;
		nt::Layer b1;
		nt::TensorGrad parameter;
		bool normalize;
		Example(int64_t hidden_features, int64_t out_features, bool normal = true)
			:l1(nt::layers::Linear(hidden_features, out_features)),
			b1(normal ? nt::Layer(nt::layers::BatchNorm1D(out_features)) : nt::Layer(nt::layers::Identity())),
			parameter(nullptr),
			normalize(normal)
			{}

		inline nt::TensorGrad forward(const nt::TensorGrad& x) override{
			nt::TensorGrad out = l1(x);
			out = b1(out);
			if(parameter.is_null()){
				parameter = nt::functional::zeros({x.shape()[-2], 1}, x.tensor.dtype);
			}
			return out + parameter;
		}

};

//this adds reflection to the layer so that gradients can be tracked properly
_NT_REGISTER_LAYER_(Example, l1, b1, parameter, normalize)
//in the above, normalize is optional to add
//any variable that is an NeuroTensor object should be included

```

