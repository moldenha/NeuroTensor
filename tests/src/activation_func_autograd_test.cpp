#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>

#define ADD_UNDERSCORE(name) name##_

inline std::ofstream& make_save_function(std::ofstream& script){
    script << "def to_neurotensor(tensor, name):\n";
    script << "\tt_n = tensor.numpy()\n";
    script << "\tnp.save(name, t_n)\n\n";
    return script;
}

inline std::ofstream& make_random_between(std::ofstream& script, std::string name, std::string shape, std::string bounds = "(-1, 1)"){
    script << "\twith torch.no_grad():\n";
    script << "\t\t"<<name<<" = torch.rand("<<shape<<").uniform_"<<bounds<<"\n";
    script << "\t"<<name<<".requires_grad_()\n";
    return script;
}

template<typename... Args>
inline bool files_exist(Args&&... args) {
    return (... && std::filesystem::exists(std::forward<Args>(args)));
}

void make_activation_function_test_torch(std::string name, bool in_functional, std::string args = "", std::string bounds="(-1, 1)"){
    std::string filename_a = "../tests/autograd_data/" + name +"_complex_input_a.npy";
    std::string filename_a2 = "../tests/autograd_data/" + name + "_complex_input_b.npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_complex_grad.npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_complex_x1grad.npy";
    std::string filename_c2 = "../tests/autograd_data/" + name + "_complex_x2grad.npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_complex_wgrad.npy";
    std::string filename_d2 = "../tests/autograd_data/" + name + "_complex_w2grad.npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_complex_bgrad.npy";
    std::string filename_e2 = "../tests/autograd_data/" + name + "_complex_b2grad.npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_complex_weight.npy";
    std::string filename_f2 = "../tests/autograd_data/" + name + "_complex_weight2.npy";
    std::string filename_g = "../tests/autograd_data/" + name + "_complex_bias.npy";
    std::string filename_g2 = "../tests/autograd_data/" + name + "_complex_bias2.npy";
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
    make_random_between(script, "x1", "3, 4, 5", bounds);    
    make_random_between(script, "w1", "3, 5", bounds);
    make_random_between(script, "b1", "3", bounds);
    make_random_between(script, "x2", "3, 4, 8", bounds);
    make_random_between(script, "w2", "3, 8", bounds);
    make_random_between(script, "b2", "3", bounds);
    script << "\to1 = torch.nn.functional.linear(x1, w1, b1)\n";
    script << "\to2 = torch.nn.functional.linear(x2, w2, b2)\n";
    if(in_functional){
        script << "\ts1 = torch.nn.functional." << name << "(o1" << args << ")\n";
        script << "\ts2 = torch.nn.functional." << name << "(o2" << args<< ")\n";
    }else{
        script << "\ts1 = torch." << name << "(o1" << args << ")\n";
        script << "\ts2 = torch." << name << "(o2" << args<< ")\n";
    }
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

void make_activation_function_test_torch_self(std::string name){
    std::string filename_a = "../tests/autograd_data/" + name +"_self_complex_input_a.npy";
    std::string filename_a2 = "../tests/autograd_data/" + name + "_self_complex_input_b.npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_self_complex_grad.npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_self_complex_x1grad.npy";
    std::string filename_c2 = "../tests/autograd_data/" + name + "_self_complex_x2grad.npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_self_complex_wgrad.npy";
    std::string filename_d2 = "../tests/autograd_data/" + name + "_self_complex_w2grad.npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_self_complex_bgrad.npy";
    std::string filename_e2 = "../tests/autograd_data/" + name + "_self_complex_b2grad.npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_self_complex_weight.npy";
    std::string filename_f2 = "../tests/autograd_data/" + name + "_self_complex_weight2.npy";
    std::string filename_g = "../tests/autograd_data/" + name + "_self_complex_bias.npy";
    std::string filename_g2 = "../tests/autograd_data/" + name + "_self_complex_bias2.npy";
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
    make_random_between(script, "x1", "3, 4, 5");    
    make_random_between(script, "w1", "3, 5");
    make_random_between(script, "b1", "3");
    make_random_between(script, "x2", "3, 4, 8");
    make_random_between(script, "w2", "3, 8");
    make_random_between(script, "b2", "3");
    script << "\to1 = torch.nn.functional.linear(x1, w1, b1)\n";
    script << "\to2 = torch.nn.functional.linear(x2, w2, b2)\n";
    script << "\ttorch." << name << "_(o1)\n";
    script << "\ttorch." << name << "_(o2)\n";
    script << "\to = o1 + o2\n";
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



#define NT_MAKE_RUN_TEST(name, in_functional, _rtol, _atol)\
    run_test(#name " autograd test", []{\
        std::string n(#name);\
        std::string filename_a = "../tests/autograd_data/" + n +"_complex_input_a.npy";\
        std::string filename_a2 = "../tests/autograd_data/" + n + "_complex_input_b.npy";\
        std::string filename_b = "../tests/autograd_data/" + n + "_complex_grad.npy";\
        std::string filename_c = "../tests/autograd_data/" + n + "_complex_x1grad.npy";\
        std::string filename_c2 = "../tests/autograd_data/" + n + "_complex_x2grad.npy";\
        std::string filename_d = "../tests/autograd_data/" + n + "_complex_wgrad.npy";\
        std::string filename_d2 = "../tests/autograd_data/" + n + "_complex_w2grad.npy";\
        std::string filename_e = "../tests/autograd_data/" + n + "_complex_bgrad.npy";\
        std::string filename_e2 = "../tests/autograd_data/" + n + "_complex_b2grad.npy";\
        std::string filename_f = "../tests/autograd_data/" + n + "_complex_weight.npy";\
        std::string filename_f2 = "../tests/autograd_data/" + n + "_complex_weight2.npy";\
        std::string filename_g = "../tests/autograd_data/" + n + "_complex_bias.npy";\
        std::string filename_g2 = "../tests/autograd_data/" + n + "_complex_bias2.npy";\
        make_activation_function_test_torch(n, in_functional);\
        nt::TensorGrad x1(nt::from_numpy(filename_a), true);\
        nt::TensorGrad x2(nt::from_numpy(filename_a2), true);\
        nt::Tensor grad = nt::from_numpy(filename_b);\
        nt::Tensor expected_x1grad = nt::from_numpy(filename_c);\
        nt::Tensor expected_x2grad = nt::from_numpy(filename_c2);\
        nt::Tensor expected_w1grad = nt::from_numpy(filename_d);\
        nt::Tensor expected_w2grad = nt::from_numpy(filename_d2);\
        nt::Tensor expected_b1grad = nt::from_numpy(filename_e);\
        nt::Tensor expected_b2grad = nt::from_numpy(filename_e2);\
        nt::TensorGrad w1(nt::from_numpy(filename_f), true);\
        nt::TensorGrad w2(nt::from_numpy(filename_f2), true);\
        nt::TensorGrad b1(nt::from_numpy(filename_g), true);\
        nt::TensorGrad b2(nt::from_numpy(filename_g2), true);\
        auto o1 = nt::linear(bias = b1, input = x1, weight = w1, transpose_b = true);\
        auto o2 = nt::linear(bias = b2, input = x2, weight = w2, transpose_b = true);\
        auto s1 = nt::name(o1);\
        auto s2 = nt::name(o2);\
        auto o = s1 + s2;\
        o.backward(grad);\
        nt::utils::throw_exception(\
             nt::allclose(x1.grad(), expected_x1grad, _rtol, _atol, true) && nt::allclose(x2.grad(), expected_x2grad, _rtol, _atol, true) && \
             nt::allclose(w1.grad(), expected_w1grad, _rtol, _atol, true) && nt::allclose(w2.grad(), expected_w2grad, _rtol, _atol, true) && \
             nt::allclose(b1.grad(), expected_b1grad, _rtol, _atol, true) && nt::allclose(b2.grad(), expected_b2grad, _rtol, _atol, true), \
            "Error, grads do not match $ \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$", \
             nt::noprintdtype,  \
             x1.grad(), expected_x1grad, x2.grad(), expected_x2grad, \
             w1.grad(), expected_w1grad, w2.grad(), expected_w2grad, \
             b1.grad(), expected_b1grad, b2.grad(), expected_b2grad);\
    });\
    // if(!in_functional){\
    // run_test(#name " - self autograd test", []{\
    //     std::string n(#name);\
    //     std::string filename_a = "../tests/autograd_data/" + n +"_complex_input_a.npy";\
    //     std::string filename_a2 = "../tests/autograd_data/" + n + "_complex_input_b.npy";\
    //     std::string filename_b = "../tests/autograd_data/" + n + "_complex_grad.npy";\
    //     std::string filename_c = "../tests/autograd_data/" + n + "_complex_x1grad.npy";\
    //     std::string filename_c2 = "../tests/autograd_data/" + n + "_complex_x2grad.npy";\
    //     std::string filename_d = "../tests/autograd_data/" + n + "_complex_wgrad.npy";\
    //     std::string filename_d2 = "../tests/autograd_data/" + n + "_complex_w2grad.npy";\
    //     std::string filename_e = "../tests/autograd_data/" + n + "_complex_bgrad.npy";\
    //     std::string filename_e2 = "../tests/autograd_data/" + n + "_complex_b2grad.npy";\
    //     std::string filename_f = "../tests/autograd_data/" + n + "_complex_weight.npy";\
    //     std::string filename_f2 = "../tests/autograd_data/" + n + "_complex_weight2.npy";\
    //     std::string filename_g = "../tests/autograd_data/" + n + "_complex_bias.npy";\
    //     std::string filename_g2 = "../tests/autograd_data/" + n + "_complex_bias2.npy";\
    //     make_activation_function_test_torch_self(n);\
    //     nt::TensorGrad x1(nt::from_numpy(filename_a), true);\
    //     nt::TensorGrad x2(nt::from_numpy(filename_a2), true);\
    //     nt::Tensor grad = nt::from_numpy(filename_b);\
    //     nt::Tensor expected_x1grad = nt::from_numpy(filename_c);\
    //     nt::Tensor expected_x2grad = nt::from_numpy(filename_c2);\
    //     nt::Tensor expected_w1grad = nt::from_numpy(filename_d);\
    //     nt::Tensor expected_w2grad = nt::from_numpy(filename_d2);\
    //     nt::Tensor expected_b1grad = nt::from_numpy(filename_e);\
    //     nt::Tensor expected_b2grad = nt::from_numpy(filename_e2);\
    //     nt::TensorGrad w1(nt::from_numpy(filename_f), true);\
    //     nt::TensorGrad w2(nt::from_numpy(filename_f2), true);\
    //     nt::TensorGrad b1(nt::from_numpy(filename_g), true);\
    //     nt::TensorGrad b2(nt::from_numpy(filename_g2), true);\
    //     auto o1 = nt::linear(bias = b1, input = x1, weight = w1, transpose_b = true);\
    //     auto o2 = nt::linear(bias = b2, input = x2, weight = w2, transpose_b = true);\
    //     nt::ADD_UNDERSCORE(name)(o1);\
    //     nt::ADD_UNDERSCORE(name)(o2);\
    //     auto o = o1 + o2;\
    //     o.backward(grad);\
    //     nt::utils::throw_exception(\
    //          nt::allclose(x1.grad(), expected_x1grad) && nt::allclose(x2.grad(), expected_x2grad) && \
    //          nt::allclose(w1.grad(), expected_w1grad) && nt::allclose(w2.grad(), expected_w2grad) && \
    //          nt::allclose(b1.grad(), expected_b1grad) && nt::allclose(b2.grad(), expected_b2grad), \
    //         "Error, grads do not match $ \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$", \
    //          nt::noprintdtype,  \
    //          x1.grad(), expected_x1grad, x2.grad(), expected_x2grad, \
    //          w1.grad(), expected_w1grad, w2.grad(), expected_w2grad, \
    //          b1.grad(), expected_b1grad, b2.grad(), expected_b2grad);\
    // });\
    // }\


void activation_test_autograd(){
    using namespace nt::literals;
    NT_MAKE_RUN_TEST(sigmoid, false, 1e-5, 1e-8)
    NT_MAKE_RUN_TEST(sqrt, false, 1e-5, 1e-8)
    // NT_MAKE_RUN_TEST(invsqrt)
    NT_MAKE_RUN_TEST(abs, false, 1e-5, 1e-8)
    NT_MAKE_RUN_TEST(relu, false, 1e-5, 1e-8)
    NT_MAKE_RUN_TEST(gelu, true, 1e-2, 1e-3)
    NT_MAKE_RUN_TEST(silu, true, 1e-3, 1e-4)
    run_test("Pow model Autograd test", []{
        make_activation_function_test_torch("pow", false, ", 3");
        std::string filename_a = "../tests/autograd_data/pow_complex_input_a.npy";
        std::string filename_a2 = "../tests/autograd_data/pow_complex_input_b.npy";
        std::string filename_b = "../tests/autograd_data/pow_complex_grad.npy";
        std::string filename_c = "../tests/autograd_data/pow_complex_x1grad.npy";
        std::string filename_c2 = "../tests/autograd_data/pow_complex_x2grad.npy";
        std::string filename_d = "../tests/autograd_data/pow_complex_wgrad.npy";
        std::string filename_d2 = "../tests/autograd_data/pow_complex_w2grad.npy";
        std::string filename_e = "../tests/autograd_data/pow_complex_bgrad.npy";
        std::string filename_e2 = "../tests/autograd_data/pow_complex_b2grad.npy";
        std::string filename_f = "../tests/autograd_data/pow_complex_weight.npy";
        std::string filename_f2 = "../tests/autograd_data/pow_complex_weight2.npy";
        std::string filename_g = "../tests/autograd_data/pow_complex_bias.npy";
        std::string filename_g2 = "../tests/autograd_data/pow_complex_bias2.npy";
        
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
        auto s1 = nt::pow(o1, exponent = 3);
        auto s2 = nt::pow(o2, exponent = 3);
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

    run_test("Softplus model Autograd test", []{
        make_activation_function_test_torch("softplus", true, ", threshold=30", "(-9, 9)");
        std::string filename_a = "../tests/autograd_data/softplus_complex_input_a.npy";
        std::string filename_a2 = "../tests/autograd_data/softplus_complex_input_b.npy";
        std::string filename_b = "../tests/autograd_data/softplus_complex_grad.npy";
        std::string filename_c = "../tests/autograd_data/softplus_complex_x1grad.npy";
        std::string filename_c2 = "../tests/autograd_data/softplus_complex_x2grad.npy";
        std::string filename_d = "../tests/autograd_data/softplus_complex_wgrad.npy";
        std::string filename_d2 = "../tests/autograd_data/softplus_complex_w2grad.npy";
        std::string filename_e = "../tests/autograd_data/softplus_complex_bgrad.npy";
        std::string filename_e2 = "../tests/autograd_data/softplus_complex_b2grad.npy";
        std::string filename_f = "../tests/autograd_data/softplus_complex_weight.npy";
        std::string filename_f2 = "../tests/autograd_data/softplus_complex_weight2.npy";
        std::string filename_g = "../tests/autograd_data/softplus_complex_bias.npy";
        std::string filename_g2 = "../tests/autograd_data/softplus_complex_bias2.npy";
        
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
        auto s1 = nt::softplus(o1, threshold = 30.0);
        auto s2 = nt::softplus(o2, threshold = 30.0);
        auto o = s1 + s2;
        float rtol = 1e-3;
        float atol = 1e-4;
        o.backward(grad);
        nt::utils::throw_exception(
             nt::allclose(x1.grad(), expected_x1grad, rtol, atol) && nt::allclose(x2.grad(), expected_x2grad, rtol, atol) &&
             nt::allclose(w1.grad(), expected_w1grad, rtol, atol) && nt::allclose(w2.grad(), expected_w2grad, rtol, atol) &&
             nt::allclose(b1.grad(), expected_b1grad, rtol, atol) && nt::allclose(b2.grad(), expected_b2grad, rtol, atol),
            "Error, grads do not match $ \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$ \n\n\n \n$ \n$ \n$ \n$",
             nt::noprintdtype, 
             x1.grad(), expected_x1grad, x2.grad(), expected_x2grad,
             w1.grad(), expected_w1grad, w2.grad(), expected_w2grad,
             b1.grad(), expected_b1grad, b2.grad(), expected_b2grad);

    });

}

#undef ADD_UNDERSCORE


// int main() {
//     conv_tests();
//     return 0;
// }
