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

void make_conv_files(std::string name, std::string inp_size, std::string weight_size, std::string args = "", std::string extension = ""){
    std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e, filename_f)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tx = torch.rand(" << inp_size << ", requires_grad=True)\n";
    script << "\tw = torch.rand(" << weight_size << ", requires_grad=True)\n";
    script << "\to = torch.nn.functional." << name << "(x, w" << args << ")\n";
    script << "\tgrad = torch.rand_like(o)\n";
    script << "\to.backward(grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_b << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(w.grad, '" << filename_d << "')\n";
    script << "\tto_neurotensor(w.detach(), '" << filename_e << "')\n";
    script << "\tto_neurotensor(o.detach(), '" << filename_f << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

// void make_linear_complex_files(){
//     std::string filename_a = "../tests/autograd_data/linear_complex_input_a.npy";
//     std::string filename_a2 = "../tests/autograd_data/linear_complex_input_b.npy";
//     std::string filename_b = "../tests/autograd_data/linear_complex_grad.npy";
//     std::string filename_c = "../tests/autograd_data/linear_complex_x1grad.npy";
//     std::string filename_c2 = "../tests/autograd_data/linear_complex_x2grad.npy";
//     std::string filename_d = "../tests/autograd_data/linear_complex_wgrad.npy";
//     std::string filename_d2 = "../tests/autograd_data/linear_complex_w2grad.npy";
//     std::string filename_e = "../tests/autograd_data/linear_complex_bgrad.npy";
//     std::string filename_e2 = "../tests/autograd_data/linear_complex_b2grad.npy";
//     std::string filename_f = "../tests/autograd_data/linear_complex_weight.npy";
//     std::string filename_f2 = "../tests/autograd_data/linear_complex_weight2.npy";
//     std::string filename_g = "../tests/autograd_data/linear_complex_bias.npy";
//     std::string filename_g2 = "../tests/autograd_data/linear_complex_bias2.npy";
//     if(files_exist(filename_a, filename_a2,
//                    filename_b, filename_c, filename_c2,
//                    filename_d, filename_d2,
//                    filename_e, filename_e2,
//                    filename_f, filename_f2,
//                    filename_g, filename_g2)) return;
//     const std::string script_name = "session.py";
//     std::ofstream script(script_name);
//     script << "# Python session\n";
//     script << "import torch\n";
//     script << "import numpy as np\n";
//     make_save_function(script);
//     script << "def func_1():\n";
//     script << "\tx1 = torch.rand(3, 4, 5, requires_grad=True)\n";
//     script << "\tw1 = torch.rand(3, 5, requires_grad=True)\n";
//     script << "\tb1 = torch.rand(3, requires_grad=True)\n";
//     script << "\tx2 = torch.rand(3, 4, 8, requires_grad=True)\n";
//     script << "\tw2 = torch.rand(3, 8, requires_grad=True)\n";
//     script << "\tb2 = torch.rand(3, requires_grad=True)\n";
//     script << "\to1 = torch.nn.functional.linear(x1, w1, b1)\n";
//     script << "\to2 = torch.nn.functional.linear(x2, w2, b2)\n";
//     script << "\ts1 = torch.nn.functional.sigmoid(o1)\n";
//     script << "\ts2 = torch.nn.functional.sigmoid(o2)\n";
//     script << "\to = s1 + s2\n";
//     script << "\tgrad = torch.rand_like(o)\n";
//     script << "\to.backward(grad)\n";
//     script << "\tto_neurotensor(x1.detach(), '" << filename_a << "')\n";
//     script << "\tto_neurotensor(x2.detach(), '" << filename_a2 << "')\n";
//     script << "\tto_neurotensor(grad, '" << filename_b << "')\n";
//     script << "\tto_neurotensor(x1.grad, '" << filename_c << "')\n";
//     script << "\tto_neurotensor(x2.grad, '" << filename_c2 << "')\n";
//     script << "\tto_neurotensor(w1.grad, '" << filename_d << "')\n";
//     script << "\tto_neurotensor(w2.grad, '" << filename_d2 << "')\n";
//     script << "\tto_neurotensor(b1.grad, '" << filename_e << "')\n";
//     script << "\tto_neurotensor(b2.grad, '" << filename_e2 << "')\n";
//     script << "\tto_neurotensor(w1.detach(), '" << filename_f << "')\n";
//     script << "\tto_neurotensor(w2.detach(), '" << filename_f2 << "')\n";
//     script << "\tto_neurotensor(b1.detach(), '" << filename_g << "')\n";
//     script << "\tto_neurotensor(b2.detach(), '" << filename_g2 << "')\n";
//     script << "\n\n";
//     script << "func_1()\n";
//     script.close();

//     std::string cmd = "python3 "+script_name;
//     std::system(cmd.c_str());
// }





void conv_autograd_test(){
    using namespace nt::literals;
    run_test("Conv1d Autograd test", []{
        std::string name = "conv1d";
        make_conv_files(name, "1, 30, 100", "80, 30, 3", ", stride = 1, padding = 1");
        std::string filename_a = "../tests/autograd_data/" + name + "_input.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad.npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight.npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output.npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv1d(x, w, stride = 1, padding = 1);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });

    run_test("Conv1d Grouped Autograd test", []{
        std::string name = "conv1d";
        std::string extension = "_grouped";
        make_conv_files(name, "1, 4, 5", "4, 2, 3", ", stride = 1, groups = 2", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv1d(x, w, stride = 1, groups = 2);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });
    run_test("Conv1d Batched Autograd test", []{
        std::string name = "conv1d";
        std::string extension = "_batched";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_conv_files(name, "10, 30, 100", "80, 30, 3", ", stride = 1, padding = 1", extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv1d(x, w, stride = 1, padding = 1);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });
    run_test("Conv1d Grouped Batched Autograd test", []{
        std::string name = "conv1d";
        std::string extension = "_grouped_batched";
        make_conv_files(name, "4, 4, 5", "4, 2, 3", ", stride = 1, groups = 2", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv1d(x, w, stride = 1, groups = 2);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));

    });
    run_test("Conv2d Autograd test", []{
        std::string name = "conv2d";
        make_conv_files(name, "1, 3, 16, 16", "4, 3, 3, 3", ", stride = 1, padding = 1");
        std::string filename_a = "../tests/autograd_data/" + name + "_input.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad.npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight.npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output.npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv2d(x, w, stride = 1, padding = 1);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });

    run_test("Conv2d Grouped Autograd test", []{
        std::string name = "conv2d";
        std::string extension = "_grouped";
        make_conv_files(name, "1, 4, 5, 5", "4, 2, 3, 3", ", stride = 1, groups = 2", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv2d(x, w, stride = 1, groups = 2);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });
    run_test("Conv2d Batched Autograd test", []{
        std::string name = "conv2d";
        std::string extension = "_batched";
        make_conv_files(name, "4, 3, 16, 16", "4, 3, 3, 3", ", stride = 1, padding = 1", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv2d(x, w, stride = 1, padding = 1);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });
    run_test("Conv2d Grouped Batched Autograd test", []{
        std::string name = "conv2d";
        std::string extension = "_grouped_batched";
        make_conv_files(name, "4, 4, 5, 5", "4, 2, 3, 3", ", stride = 1, groups = 2", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv2d(x, w, stride = 1, groups = 2);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });

    run_test("Conv3d Autograd test", []{
        std::string name = "conv3d";
        make_conv_files(name, "1, 3, 16, 16, 16", "4, 3, 3, 3, 3", ", stride = 1, padding = 1");
        std::string filename_a = "../tests/autograd_data/" + name + "_input.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad.npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight.npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output.npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv3d(x, w, stride = 1, padding = 1);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match ",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match ",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });

    run_test("Conv3d Grouped Autograd test", []{
        std::string name = "conv3d";
        std::string extension = "_grouped";
        make_conv_files(name, "1, 4, 5, 5, 5", "4, 2, 3, 3, 3", ", stride = 1, groups = 2", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv3d(x, w, stride = 1, groups = 2);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });
    run_test("Conv3d Batched Autograd test", []{
        std::string name = "conv3d";
        std::string extension = "_batched";
        make_conv_files(name, "4, 3, 16, 16, 16", "4, 3, 3, 3, 3", ", stride = 1, padding = 1", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv3d(x, w, stride = 1, padding = 1);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });
    run_test("Conv3d Grouped Batched Autograd test", []{
        std::string name = "conv3d";
        std::string extension = "_grouped_batched";
        make_conv_files(name, "4, 4, 5, 5, 5", "4, 2, 3, 3, 3", ", stride = 1, groups = 2", extension);
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad" + extension + ".npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight" + extension + ".npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv3d(x, w, stride = 1, groups = 2);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });

    run_test("ConvND (N = 4) Autograd test", []{
        nt::TensorGrad x(nt::rand(0, 3, {1, 3, 16, 16, 16, 16}), true);
        nt::TensorGrad w(nt::rand(0, 3, {4, 3, 3, 3, 3, 3}), true);
        auto out = nt::convnd(x, w, stride = 1, padding = 1, dim = 4);
        nt::Tensor grad = nt::rand(0, 1, out.shape());
        out.backward(grad);
        nt::utils::throw_exception(!nt::all(x.grad() == 0),
                               "Error, x grad not created for ConvND autograd test");
        nt::utils::throw_exception(!nt::all(w.grad() == 0),
                               "Error, w grad not created for ConvND autograd test"); 
    });

    run_test("ConvND (N = 4) Grouped Autograd test", []{
        nt::TensorGrad x(nt::rand(0, 3, {1, 4, 5, 5, 5, 5}), true);
        nt::TensorGrad w(nt::rand(0, 3, {4, 2, 3, 3, 3, 3}), true);
        auto out = nt::convnd(x, w, stride = 1, groups = 2, dim = 4);
        nt::Tensor grad = nt::rand(0, 1, out.shape());
        out.backward(grad);
        nt::utils::throw_exception(!nt::all(x.grad() == 0),
                               "Error, x grad not created for ConvND autograd test");
        nt::utils::throw_exception(!nt::all(w.grad() == 0),
                               "Error, w grad not created for ConvND autograd test");

    });

    run_test("ConvND (N = 4) Batched Autograd test", []{
        nt::TensorGrad x(nt::rand(0, 3, {4, 3, 16, 16, 16, 16}), true);
        nt::TensorGrad w(nt::rand(0, 3, {4, 3, 3, 3, 3, 3}), true);
        auto out = nt::convnd(x, w, stride = 1, padding = 1, dim = 4);
        nt::Tensor grad = nt::rand(0, 1, out.shape());
        out.backward(grad);
        nt::utils::throw_exception(!nt::all(x.grad() == 0),
                               "Error, x grad not created for ConvND autograd test");
        nt::utils::throw_exception(!nt::all(w.grad() == 0),
                               "Error, w grad not created for ConvND autograd test"); 
    });


    run_test("ConvND (N = 4) Grouped Batched Autograd test", []{
        nt::TensorGrad x(nt::rand(0, 3, {4, 4, 5, 5, 5, 5}), true);
        nt::TensorGrad w(nt::rand(0, 3, {4, 2, 3, 3, 3, 3}), true);
        auto out = nt::convnd(x, w, stride = 1, groups = 2, dim = 4);
        nt::Tensor grad = nt::rand(0, 1, out.shape());
        out.backward(grad);
        nt::utils::throw_exception(!nt::all(x.grad() == 0),
                               "Error, x grad not created for ConvND autograd test");
        nt::utils::throw_exception(!nt::all(w.grad() == 0),
                               "Error, w grad not created for ConvND autograd test");

    });


    run_test("Conv1d Transpose Autograd test", []{
        std::string name = "conv_transpose1d";
        make_conv_files(name, "1, 3, 7", "3, 8, 3", ", stride = 2, output_padding = 1");
        std::string filename_a = "../tests/autograd_data/" + name + "_input.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_wgrad.npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_weight.npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output.npy";
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::TensorGrad w(nt::from_numpy(filename_e), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_wgrad = nt::from_numpy(filename_d);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        nt::TensorGrad o = nt::conv_transpose1d(x, w, stride = 2, output_padding = 1);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        std::cout << "running backward now..." << std::endl;
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
        nt::utils::throw_exception(
            nt::allclose(w.grad(), expected_wgrad),
            "error: gradients for w do not match $ \n$ \n$ \n$",
            nt::noprintdtype, w.grad(), expected_wgrad, nt::isclose(w.grad(), expected_wgrad));
    });

    
}

