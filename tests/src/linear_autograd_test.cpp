#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>

inline std::ofstream& make_save_function(std::ofstream& script){
    script << "def to_neurotensor(tensor, name):\n";
    script << "\tt_n = tensor.numpy()\n";
    script << "\tnp.save(name, t_n)\n\n";
    return script;
}

template<typename... Args>
inline bool files_exist(Args&&... args) {
    return (... && std::filesystem::exists(std::forward<Args>(args)));
}

void make_linear_files(){
    std::string filename_a = "../tests/autograd_data/linear_input.npy";
    std::string filename_b = "../tests/autograd_data/linear_grad.npy";
    std::string filename_c = "../tests/autograd_data/linear_xgrad.npy";
    std::string filename_d = "../tests/autograd_data/linear_wgrad.npy";
    std::string filename_e = "../tests/autograd_data/linear_bgrad.npy";
    std::string filename_f = "../tests/autograd_data/linear_weight.npy";
    std::string filename_g = "../tests/autograd_data/linear_bias.npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx = torch.rand(3, 4, 5, requires_grad=True)\n";
    script << "\tw = torch.rand(3, 5, requires_grad=True)\n";
    script << "\tb = torch.rand(3, requires_grad=True)\n";
    script << "\to = torch.nn.functional.linear(x, w, b)\n";
    script << "\tgrad = torch.rand_like(o)\n";
    script << "\to.backward(grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_b << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(w.grad, '" << filename_d << "')\n";
    script << "\tto_neurotensor(b.grad, '" << filename_e << "')\n";
    script << "\tto_neurotensor(w.detach(), '" << filename_f << "')\n";
    script << "\tto_neurotensor(b.detach(), '" << filename_g << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

void make_linear_complex_files(){
    std::string filename_a = "../tests/autograd_data/linear_complex_input_a.npy";
    std::string filename_a2 = "../tests/autograd_data/linear_complex_input_b.npy";
    std::string filename_b = "../tests/autograd_data/linear_complex_grad.npy";
    std::string filename_c = "../tests/autograd_data/linear_complex_x1grad.npy";
    std::string filename_c2 = "../tests/autograd_data/linear_complex_x2grad.npy";
    std::string filename_d = "../tests/autograd_data/linear_complex_wgrad.npy";
    std::string filename_d2 = "../tests/autograd_data/linear_complex_w2grad.npy";
    std::string filename_e = "../tests/autograd_data/linear_complex_bgrad.npy";
    std::string filename_e2 = "../tests/autograd_data/linear_complex_b2grad.npy";
    std::string filename_f = "../tests/autograd_data/linear_complex_weight.npy";
    std::string filename_f2 = "../tests/autograd_data/linear_complex_weight2.npy";
    std::string filename_g = "../tests/autograd_data/linear_complex_bias.npy";
    std::string filename_g2 = "../tests/autograd_data/linear_complex_bias2.npy";
    if(files_exist(filename_a, filename_a2,
                   filename_b, filename_c, filename_c2,
                   filename_d, filename_d2,
                   filename_e, filename_e2,
                   filename_f, filename_f2,
                   filename_g, filename_g2)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx1 = torch.rand(3, 4, 5, requires_grad=True)\n";
    script << "\tw1 = torch.rand(3, 5, requires_grad=True)\n";
    script << "\tb1 = torch.rand(3, requires_grad=True)\n";
    script << "\tx2 = torch.rand(3, 4, 8, requires_grad=True)\n";
    script << "\tw2 = torch.rand(3, 8, requires_grad=True)\n";
    script << "\tb2 = torch.rand(3, requires_grad=True)\n";
    script << "\to1 = torch.nn.functional.linear(x1, w1, b1)\n";
    script << "\to2 = torch.nn.functional.linear(x2, w2, b2)\n";
    script << "\ts1 = torch.nn.functional.sigmoid(o1)\n";
    script << "\ts2 = torch.nn.functional.sigmoid(o2)\n";
    script << "\to = s1 + s2\n";
    script << "\tgrad = torch.rand_like(o)\n";
    script << "\to.backward(grad)\n";
    script << "\tto_neurotensor(x1.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(x2.detach(), '" << filename_a2 << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_b << "')\n";
    script << "\tto_neurotensor(x1.grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(x2.grad, '" << filename_c2 << "')\n";
    script << "\tto_neurotensor(w1.grad, '" << filename_d << "')\n";
    script << "\tto_neurotensor(w2.grad, '" << filename_d2 << "')\n";
    script << "\tto_neurotensor(b1.grad, '" << filename_e << "')\n";
    script << "\tto_neurotensor(b2.grad, '" << filename_e2 << "')\n";
    script << "\tto_neurotensor(w1.detach(), '" << filename_f << "')\n";
    script << "\tto_neurotensor(w2.detach(), '" << filename_f2 << "')\n";
    script << "\tto_neurotensor(b1.detach(), '" << filename_g << "')\n";
    script << "\tto_neurotensor(b2.detach(), '" << filename_g2 << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}





void linear_autograd_test(){
    using namespace nt::literals;
    run_test("Linear Autograd test", []{
        make_linear_files();
        nt::TensorGrad x(nt::from_numpy("../tests/autograd_data/linear_input.npy"), true);
        nt::Tensor grad = nt::from_numpy("../tests/autograd_data/linear_grad.npy");
        nt::Tensor expected_xgrad = nt::from_numpy("../tests/autograd_data/linear_xgrad.npy");
        nt::Tensor expected_wgrad = nt::from_numpy("../tests/autograd_data/linear_wgrad.npy");
        nt::Tensor expected_bgrad = nt::from_numpy("../tests/autograd_data/linear_bgrad.npy");
        nt::TensorGrad w(nt::from_numpy("../tests/autograd_data/linear_weight.npy"), true);
        nt::TensorGrad b(nt::from_numpy("../tests/autograd_data/linear_bias.npy"), true);
        nt::TensorGrad o = nt::linear(bias = b, input = x, weight = w, transpose_b = true);
        o.backward(grad);
        nt::utils::throw_exception(
             nt::allclose(x.grad(), expected_xgrad) && 
             nt::allclose(w.grad(), expected_wgrad) && 
             nt::allclose(b.grad(), expected_bgrad),
            "Error, grads do not match $ \n$ \n$ \n\n\n \n$ \n$ \n\n\n \n$ \n$",
             nt::noprintdtype, x.grad(), expected_xgrad,
             w.grad(), expected_wgrad,
             b.grad(), expected_bgrad);

    });

    run_test("Linear Complex Autograd test", []{
        make_linear_complex_files();
        std::string filename_a = "../tests/autograd_data/linear_complex_input_a.npy";
        std::string filename_a2 = "../tests/autograd_data/linear_complex_input_b.npy";
        std::string filename_b = "../tests/autograd_data/linear_complex_grad.npy";
        std::string filename_c = "../tests/autograd_data/linear_complex_x1grad.npy";
        std::string filename_c2 = "../tests/autograd_data/linear_complex_x2grad.npy";
        std::string filename_d = "../tests/autograd_data/linear_complex_wgrad.npy";
        std::string filename_d2 = "../tests/autograd_data/linear_complex_w2grad.npy";
        std::string filename_e = "../tests/autograd_data/linear_complex_bgrad.npy";
        std::string filename_e2 = "../tests/autograd_data/linear_complex_b2grad.npy";
        std::string filename_f = "../tests/autograd_data/linear_complex_weight.npy";
        std::string filename_f2 = "../tests/autograd_data/linear_complex_weight2.npy";
        std::string filename_g = "../tests/autograd_data/linear_complex_bias.npy";
        std::string filename_g2 = "../tests/autograd_data/linear_complex_bias2.npy";
        
        nt::TensorGrad x1(nt::from_numpy(filename_a), true);
        nt::TensorGrad x2(nt::from_numpy(filename_a2), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_x1grad = nt::from_numpy(filename_c);
        nt::Tensor expected_x2grad = nt::from_numpy(filename_c2);
        nt::Tensor expected_w1grad = nt::from_numpy(filename_d);
        nt::Tensor expected_w2grad = nt::from_numpy(filename_d2);
        nt::Tensor expected_b1grad = nt::from_numpy(filename_e);
        nt::Tensor expected_b2grad = nt::from_numpy(filename_e2);
        nt::TensorGrad w1(nt::from_numpy(filename_f), true);
        nt::TensorGrad w2(nt::from_numpy(filename_f2), true);
        nt::TensorGrad b1(nt::from_numpy(filename_g), true);
        nt::TensorGrad b2(nt::from_numpy(filename_g2), true);
    
        auto o1 = nt::linear(bias = b1, input = x1, weight = w1, transpose_b = true);
        auto o2 = nt::linear(bias = b2, input = x2, weight = w2, transpose_b = true);
        auto s1 = nt::sigmoid(o1);
        auto s2 = nt::sigmoid(o2);
        auto o = s1 + s2;
        o.backward(grad);
        nt::utils::throw_exception(
             nt::allclose(x1.grad(), expected_x1grad) && nt::allclose(x2.grad(), expected_x2grad) &&
             nt::allclose(w1.grad(), expected_w1grad) && nt::allclose(w2.grad(), expected_w2grad) &&
             nt::allclose(b1.grad(), expected_b1grad) && nt::allclose(b2.grad(), expected_b2grad),
            "Error, grads do not match $ \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$",
             nt::noprintdtype, 
             x1.grad(), expected_x1grad, x2.grad(), expected_x2grad,
             w1.grad(), expected_w1grad, w2.grad(), expected_w2grad,
             b1.grad(), expected_b1grad, b2.grad(), expected_b2grad);

    });
}

