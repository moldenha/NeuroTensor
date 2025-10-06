#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include <nt/convert/Convert.h>
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

void make_view_files(std::string name, std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_inputa" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_inputb" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_agrad" + extension + ".npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_bgrad" + extension + ".npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e, filename_f)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\ta = torch.rand(4, 5, requires_grad = True)\n";
    script << "\tb = torch.rand(5, requires_grad = True)\n";
    script << "\tc = a + b\n";
    script << "\tc *= 10\n";
    script << "\td = c / 3\n";
    script << "\ty = d[1]\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(gradient=grad)\n";
    script << "\tto_neurotensor(a.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(b.detach(), '" << filename_b << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(a.grad, '" << filename_d << "')\n";
    script << "\tto_neurotensor(b.grad, '" << filename_e << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_f << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

void make_view_fuller_self_files(std::string name, std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_inputa" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_inputb" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_agrad" + extension + ".npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_bgrad" + extension + ".npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e, filename_f)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\ta = torch.rand(4, 5, requires_grad = True)\n";
    script << "\tb = torch.rand(5, requires_grad = True)\n";
    script << "\tc = a + b\n";
    script << "\tc *= 10\n";
    script << "\tc += 99\n";
    script << "\td = c / 3\n";
    script << "\ty = d[1]\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(gradient=grad)\n";
    script << "\tto_neurotensor(a.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(b.detach(), '" << filename_b << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(a.grad, '" << filename_d << "')\n";
    script << "\tto_neurotensor(b.grad, '" << filename_e << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_f << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

void view_autograd_test(){
    using namespace nt::literals;
    // flip test
    run_test("view last- Autograd test", [] {
        std::string name = "view_last";
        std::string extension = "";
        std::string args = "";
        std::string filename_a = "../tests/autograd_data/" + name + "_inputa" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_inputb" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_agrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_bgrad" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_view_files(name, args, extension);
        nt::TensorGrad a(nt::from_numpy(filename_a), true);
        nt::TensorGrad b(nt::from_numpy(filename_b), true);
        nt::Tensor grad = nt::from_numpy(filename_c);
        nt::Tensor expected_agrad = nt::from_numpy(filename_d);
        nt::Tensor expected_bgrad = nt::from_numpy(filename_e);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        auto c = a + b;
        // auto m = c * 10;
        c *= 10;
        auto d = c / 3;
        auto y = d[1];
        auto auto_grad = y.get_auto_grad();
        auto path = auto_grad.get_path();
        // for(const auto& Node : path)
        //     std::cout << Node->backwardFunc->get_name() << "->";
        // std::cout << "Done"<<std::endl;
        y.backward(grad);

        nt::utils::throw_exception(
            nt::allclose(y.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, y.detach(), expected_output, nt::isclose(y.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(a.grad(), expected_agrad),
            "error: gradients for a do not match $ \n$ \n$ \n$",
            nt::noprintdtype, a.grad(), expected_agrad, nt::isclose(a.grad(), expected_agrad));
        nt::utils::throw_exception(
            nt::allclose(b.grad(), expected_bgrad),
            "error: gradients for a do not match $ \n$ \n$ \n$",
            nt::noprintdtype, b.grad(), expected_bgrad, nt::isclose(b.grad(), expected_bgrad));
    });

    run_test("view last (fuller self) - Autograd test", [] {
        std::string name = "view_last_fuller_self";
        std::string extension = "";
        std::string args = "";
        std::string filename_a = "../tests/autograd_data/" + name + "_inputa" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_inputb" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_agrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_bgrad" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_view_fuller_self_files(name, args, extension);
        nt::TensorGrad a(nt::from_numpy(filename_a), true);
        nt::TensorGrad b(nt::from_numpy(filename_b), true);
        nt::Tensor grad = nt::from_numpy(filename_c);
        nt::Tensor expected_agrad = nt::from_numpy(filename_d);
        nt::Tensor expected_bgrad = nt::from_numpy(filename_e);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        auto c = a + b;
        // auto m = c * 10;
        c *= 10;
        c += 99;
        auto d = c / 3;
        auto y = d[1];
        auto auto_grad = y.get_auto_grad();
        auto path = auto_grad.get_path();
        // for(const auto& Node : path)
        //     std::cout << Node->backwardFunc->get_name() << "->";
        // std::cout << "Done"<<std::endl;
        y.backward(grad);

        nt::utils::throw_exception(
            nt::allclose(y.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, y.detach(), expected_output, nt::isclose(y.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(a.grad(), expected_agrad),
            "error: gradients for a do not match $ \n$ \n$ \n$",
            nt::noprintdtype, a.grad(), expected_agrad, nt::isclose(a.grad(), expected_agrad));
        nt::utils::throw_exception(
            nt::allclose(b.grad(), expected_bgrad),
            "error: gradients for a do not match $ \n$ \n$ \n$",
            nt::noprintdtype, b.grad(), expected_bgrad, nt::isclose(b.grad(), expected_bgrad));
    });
}

