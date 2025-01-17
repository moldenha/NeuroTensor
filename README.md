![github_logo](https://github.com/user-attachments/assets/19861561-d196-4b37-b849-edf2b9a290a6)
<hr>

Version: v0.1.0

NeuroTensor is a tensor framework that has a tda and autograd wrapper built in. The syntax follows PyTorch's closely, but in C++. Current version is 0.1.0 and there is still much to come for added support and testing. This is an early beta version, with an end goal being a framework for computational neuroscience tools.


## Tensors and NeuroTensor

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

## Creating Custom Layers

There is an `nt::Layer` class with a built-in reflection wrapper. Each custom layer or model must be a class that inherits from `nt::Module`. Gradients are automatically tracked, but can be manually created as well by overloading the `backward` function. The `eval` function can also be overloaded if there are differences in the way that the layer or model behaves based on if it is in `eval` mode beyond just not tracking the gradient. 

The idea behind the API was to make it as simple and flexible as possible, while also allowing some of the principles behind c++ to shine through. Below is an example involving as many or as few arguments as needed, and showing the ability to use things such as references. Of course pointers are also allowed. Currently, default parameters and named parameters are not supported. Support for default parameters are currently being added. 

Below is an example making and usign custom layers:

```C++

#include <nt/ai.h>
#include <nt/Tensor.h>


class TimeAwareHiddenLayer : public nt::Module{
    int64_t _input_size, _hidden_size;
    public:
        nt::Layer input_proj, hidden_proj, time_proj, output_proj;
        TimeAwareHiddenLayer(int64_t input_size, int64_t hidden_size)
        :_input_size(input_size),
        _hidden_size(hidden_size),
        input_proj(nt::layers::Linear(input_size, hidden_size)),
        hidden_proj(nt::layers::Linear(hidden_size, hidden_size)),
        time_proj(nt::layers::Linear(1, hidden_size)) // Scalar time which is going to be a tensor of shape (1,1)
        output_proj(nt::layers::Linear(hidden_size, hidden_size))
        {}

        // Automatically tracks if references are expected
        // For the nt::Scalar, if a scalar rvalue is passed, will automatically convert it to an nt::Scalar
        nt::TensorGrad forward(const nt::TensorGrad& x, nt::TensorGrad& h, nt::Scalar t){
            // Params: x: input (const reference)
            //         h: reference to hidden hidden
            //         t: time
            nt::TensorGad time(nt::Tensor(t).view(1,1), false); // The layer class expects an nt::TensorGrad argument
            
            nt::TensorGrad x_proj = this->input_proj(x);
            nt::TensorGrad h_proj = this->hidden_proj(h);
            nt::TensorGrad t_proj = this->time_proj(time);
            
            // Will modify the value the reference h points to
            if(t.get<int64_t>() == 1){
                h += h_proj;
            }
            else{
                h -= h_proj;
            }
            nt::TensorGrad combined = nt::functional::relu(x_proj + h_proj + t_proj);
            nt::TensorGrad output = this->output_proj(combined);
            return std::move(output);
        }
};


class WrapperLayer : public nt::Module{
    int64_t _hidden_size;
    public:
        nt::Layer time_aware_layer;
        nt::TensorGrad default_hidden; // Stores a learnable default hidden state
        WrapperLayer(int64_t input_size, int64_t hidden_size)
        :_hidden_size(hidden_size),
        time_aware_layer(TimeAwareHiddenLayer(input_size,  hidden_size)),
        default_hidden(nt::functional::zeros({1, hidden_size}))
        {}
        
        nt::TensorGrad forward(nt::TensorGrad x){
            int64_t batch_size = x.shape()[0];
            nt::TensorGrad h = this->default_hidden.repeat(batch_size);
            // will convert 1 to an nt::Scalar
            // and use the lvalue references of x and h
            return this->time_aware_layer(x, h, 1); //will convert 1 to an nt::Scalar and 
        }
        

};



//this adds reflection to the layer so that gradients can be tracked properly and automatically
_NT_REGISTER_LAYER_(TimeAwareHiddenLayer, input_proj, hidden_proj, time_proj, output_proj)
_NT_REGISTER_LAYER_(WrapperLayer, time_aware_layer, default_hidden)
//any variable that is a NeuroTensor object should be included


int main(){
    auto critereon = nt::loss::MSE;
	nt::TensorGrad input(nt::functional::randn({30, 20}, nt::DType::Float32));
	nt::Tensor wanted = nt::functional::randint(0, 1, {30, 10}).to(nt::DType::Float32);
	nt::Layer model = WrapperLayer(20, 10);
	nt::optimizers::Adam optimizer(model.parameters(), 0.01);
	optimizer.zero_grad();
    
    //training
	for(int64_t i = 0; i < 10; ++i){
		nt::TensorGrad output = model(input);
		nt::TensorGrad loss = critereon(output, wanted);
		std::cout << "loss: "<<loss.item() << std::endl;
		loss.backward();
		optimizer.step();
	}
	std::cout << model(input) << std::endl;
    std::cout << wanted << std::endl;
    return 0;
}

```

## Autograd Usage

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


