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

void make_normalize_files(std::string name, std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_running_mean" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_running_var" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_bias" + extension + ".npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_g = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
    std::string filename_h = "../tests/autograd_data/" + name + "_weight_grad" + extension + ".npy";
    std::string filename_i = "../tests/autograd_data/" + name + "_bias_grad" + extension + ".npy";
    std::string filename_j = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e,
                   filename_f, filename_g, filename_h, filename_i, filename_j)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx = torch.rand(4, 5, 3, 8, requires_grad = True)\n";
    script << "\trunning_mean = torch.rand(5, requires_grad = False)\n";
    script << "\trunning_var = torch.rand(5, requires_grad = False)\n";
    script << "\toriginal_running_mean = running_mean.detach().clone()\n";
    script << "\toriginal_running_var = running_var.detach().clone()\n";
    script << "\tweight = torch.rand(5, requires_grad = True)\n";
    script << "\tbias = torch.rand(5, requires_grad = True)\n";
    script << "\ty = torch.nn.functional.batch_norm(x, running_mean, running_var = running_var, weight = weight, bias = bias, training = True)\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(original_running_mean.detach(), '" << filename_b << "')\n";
    script << "\tto_neurotensor(original_running_var.detach(), '" << filename_c << "')\n";
    script << "\tto_neurotensor(weight.detach(), '" << filename_d << "')\n";
    script << "\tto_neurotensor(bias.detach(), '" << filename_e << "')\n";
    script << "\tto_neurotensor(grad.detach(), '" << filename_f << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_g << "')\n";
    script << "\tto_neurotensor(weight.grad, '" << filename_h << "')\n";
    script << "\tto_neurotensor(bias.grad, '" << filename_i << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_j << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

void make_group_normalize_files(std::string name, std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_bias" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_weight_grad" + extension + ".npy";
    std::string filename_g = "../tests/autograd_data/" + name + "_bias_grad" + extension + ".npy";
    std::string filename_h = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e,
                   filename_f, filename_g, filename_h)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx = torch.rand(3, 16, 3, 8, requires_grad = True)\n";
    script << "\tweight = torch.rand(16, requires_grad = True)\n";
    script << "\tbias = torch.rand(16, requires_grad = True)\n";
    script << "\ty = torch.nn.functional.group_norm(x, 4, weight = weight, bias = bias, eps = 1e-05)\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(weight.detach(), '" << filename_b << "')\n";
    script << "\tto_neurotensor(bias.detach(), '" << filename_c << "')\n";
    script << "\tto_neurotensor(grad.detach(), '" << filename_d << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_e << "')\n";
    script << "\tto_neurotensor(weight.grad, '" << filename_f << "')\n";
    script << "\tto_neurotensor(bias.grad, '" << filename_g << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_h << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}


inline void reproduce(nt::utils::optional_tensor_variant vtg = nullptr){
    std::cout << "function called" << std::endl;
    nt::utils::optional_tensorgrad o = nt::functional::details::force_optional_tg(vtg); 
    if(bool(o)){
        std::cout << o.value().numel() << std::endl;
    }
}


void normalize_autograd_test(){
    using namespace nt::literals;
    run_test("batch_norm - Autograd (training = true)", [] {
        std::string name = "batch_norm";
        std::string extension = "";
        std::string args = "";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_running_mean" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_running_var" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_bias" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_g = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_h = "../tests/autograd_data/" + name + "_weight_grad" + extension + ".npy";
        std::string filename_i = "../tests/autograd_data/" + name + "_bias_grad" + extension + ".npy";
        std::string filename_j = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_normalize_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor running_mean = nt::from_numpy(filename_b);
        nt::Tensor running_var = nt::from_numpy(filename_c);
        nt::TensorGrad weight_(nt::from_numpy(filename_d), true);
        nt::TensorGrad bias_(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_f);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_g);
        nt::Tensor expected_weight_grad = nt::from_numpy(filename_h);
        nt::Tensor expected_bias_grad = nt::from_numpy(filename_i);
        nt::Tensor expected_output = nt::from_numpy(filename_j);
        auto o = nt::batch_norm(x, running_mean, running_var, weight = weight_, training = true, bias = bias_);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o, expected_output, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o, expected_output, nt::isclose(o, expected_output));
        
        nt::utils::throw_exception(
            nt::allclose(bias_.grad(), expected_bias_grad, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: bias gradients do not match $ \n$ \n$ \n$",
            nt::noprintdtype, bias_.grad(), expected_bias_grad, nt::isclose(bias_.grad(), expected_bias_grad));

        nt::utils::throw_exception(
            nt::allclose(weight_.grad(), expected_weight_grad, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: weight gradients do not match $ \n$ \n$ \n$",
            nt::noprintdtype, weight_.grad(), expected_weight_grad, nt::isclose(weight_.grad(), expected_weight_grad));
        
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: x gradients do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        // nt::utils::optional_tensorgrad tga = nt::TensorGrad(nt::randn({4, 5, 6}), true);
        // nt::utils::optional_tensorgrad tgb = nullptr;
        // nt::utils::optional_tensor ta = nt::Tensor({200}, nt::DType::Float32);
        // nt::utils::optional_tensor tb = nullptr;
        // nt::utils::optional_tensor_variant var = nt::TensorGrad(nt::randn({20}));
        // // tgb = nt::functional::details::force_optional_tg(var);
        // std::cout << "made ex" << std::endl;
        // nt::TensorGrad ex(nt::randn({20}));
        // std::cout << "reproducing with ex" << std::endl;
        // reproduce(ex);
        // if(var.tracking_grad()){
        //     tgb = nt::utils::optional_tensorgrad(var);
        // }else{
        //     tb = nt::utils::optional_tensor(var);
        // }
        // tgb.reset();
    });

    run_test("group_norm - Autograd", [] {
        std::string name = "group_norm";
        std::string extension = "";
        std::string args = "";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_bias" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_weight_grad" + extension + ".npy";
        std::string filename_g = "../tests/autograd_data/" + name + "_bias_grad" + extension + ".npy";
        std::string filename_h = "../tests/autograd_data/" + name + "_output" + extension + ".npy";

        make_group_normalize_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad weight_(nt::from_numpy(filename_b), true);
        nt::TensorGrad bias_(nt::from_numpy(filename_c), true);
        nt::Tensor grad = nt::from_numpy(filename_d);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_e);
        nt::Tensor expected_weight_grad = nt::from_numpy(filename_f);
        nt::Tensor expected_bias_grad = nt::from_numpy(filename_g);
        nt::Tensor expected_output = nt::from_numpy(filename_h);
        auto o = nt::group_norm(x, num_groups = 4, weight = weight_, bias = bias_, eps = 1e-05);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o, expected_output, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o, expected_output, nt::isclose(o, expected_output));
        
        nt::utils::throw_exception(
            nt::allclose(bias_.grad(), expected_bias_grad, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: bias gradients do not match $ \n$ \n$ \n$",
            nt::noprintdtype, bias_.grad(), expected_bias_grad, nt::isclose(bias_.grad(), expected_bias_grad));

        nt::utils::throw_exception(
            nt::allclose(weight_.grad(), expected_weight_grad, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: weight gradients do not match $ \n$ \n$ \n$",
            nt::noprintdtype, weight_.grad(), expected_weight_grad, nt::isclose(weight_.grad(), expected_weight_grad));

        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad, rtol = 1e-4, ntarg_(atol) = 1e-5),
            "Error: x gradients do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));


        

    });

}


