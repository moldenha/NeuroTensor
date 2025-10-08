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

void make_operator_scalar_files(std::string name, std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx = torch.rand(4, 5, requires_grad = True)\n";
    script << "\ty = torch."<<name<<"(x, 5)\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(gradient=grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_b << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_d << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}


void make_operator_files(std::string name, std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_inputb" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_bgrad" + extension + ".npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    // std::cout << "make operator files called, and filenames are "<<filename_a<<" through "<<filename_f<<std::endl;
    // std::cout << std::boolalpha << 
    //                 files_exist(filename_a) << ' ' << 
    //                 files_exist(filename_b) << ' ' << 
    //                 files_exist(filename_c) << ' ' << 
    //                 files_exist(filename_d) << ' ' << 
    //                 files_exist(filename_e) << ' ' << 
    //                 files_exist(filename_f) << ' ' << 
    //                 files_exist(filename_a, filename_b, filename_c, filename_d, filename_e, filename_f) << std::noboolalpha << std::endl;
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e, filename_f)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx = torch.rand(4, 5, requires_grad = True)\n";
    script << "\tb = torch.rand(1, 5, requires_grad = True)\n";
    script << "\ty = torch."<<name<<"(x, b)\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(gradient=grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(b.detach(), '" << filename_b << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_d << "')\n";
    script << "\tto_neurotensor(b.grad, '" << filename_e << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_f << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

void make_reciprical_files(std::string name = "reciprocal", std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx = torch.rand(4, 5, requires_grad = True)\n";
    script << "\ty = torch."<<name<<"(x)\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(gradient=grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_b << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_d << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

#define NT_MAKE_OPERATOR_TEST(name, op)\
    run_test(#name " scalar Autograd", [] {\
        std::string name = #name;\
        std::string extension = "_scalar";\
        std::string args = "";\
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";\
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";\
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";\
        std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";\
        make_operator_scalar_files(name, args, extension);\
        nt::TensorGrad x(nt::from_numpy(filename_a), true);\
        nt::Tensor grad = nt::from_numpy(filename_b);\
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);\
        nt::Tensor expected_output = nt::from_numpy(filename_d);\
        auto o = nt::name(x, 5);\
        o.backward(grad);\
        nt::utils::throw_exception(\
            nt::allclose(o.detach(), expected_output),\
            "Error: outputs do not match $ \n$ \n$ \n$",\
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));\
        nt::utils::throw_exception(\
            nt::allclose(x.grad(), expected_xgrad),\
            "error: gradients for x do not match $ \n$ \n$ \n$",\
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));\
    });\
    run_test(#name " tensors Autograd", []{\
        std::string name = #name;\
        std::string extension = "";\
        std::string args = "";\
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";\
        std::string filename_b = "../tests/autograd_data/" + name + "_inputb" + extension + ".npy";\
        std::string filename_c = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";\
        std::string filename_d = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";\
        std::string filename_e = "../tests/autograd_data/" + name + "_bgrad" + extension + ".npy";\
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";\
        make_operator_files(name, args, extension);\
        nt::TensorGrad x(nt::from_numpy(filename_a), true);\
        nt::TensorGrad b(nt::from_numpy(filename_b), true);\
        nt::Tensor grad = nt::from_numpy(filename_c);\
        nt::Tensor expected_xgrad = nt::from_numpy(filename_d);\
        nt::Tensor expected_bgrad = nt::from_numpy(filename_e);\
        nt::Tensor expected_output = nt::from_numpy(filename_f);\
        auto o = nt::name(x, b);\
        o.backward(grad);\
        nt::utils::throw_exception(\
            nt::allclose(o.detach(), expected_output),\
            "Error: outputs do not match $ \n$ \n$ \n$",\
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));\
        nt::utils::throw_exception(\
            nt::allclose(x.grad(), expected_xgrad),\
            "error: gradients for x do not match $ \n$ \n$ \n$",\
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));\
        nt::utils::throw_exception(\
            nt::allclose(b.grad(), expected_bgrad),\
            "error: gradients for b do not match $ \n$ \n$ \n$",\
            nt::noprintdtype, b.grad(), expected_bgrad, nt::isclose(b.grad(), expected_bgrad));\
    });


void operator_autograd_test(){
    using namespace nt::literals;
    NT_MAKE_OPERATOR_TEST(multiply, *);
    NT_MAKE_OPERATOR_TEST(add, +);
    NT_MAKE_OPERATOR_TEST(subtract, -);
    NT_MAKE_OPERATOR_TEST(divide, /);
    NT_MAKE_OPERATOR_TEST(remainder, %);
    NT_MAKE_OPERATOR_TEST(fmod, %);
    run_test("inverse - reciprocal Autograd", [] {
        std::string name = "reciprocal";
        std::string extension = "";
        std::string args = "";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_reciprical_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_output = nt::from_numpy(filename_d);
        auto o = nt::inverse(x);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
    });

}


#undef NT_MAKE_OPERATOR_TEST 
