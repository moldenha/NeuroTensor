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

void make_min_max_files(std::string name, std::string args = "", std::string extension = ""){
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
    script << "\tx = torch.tensor([[1.0, 1.0, 0.0], [2.0, 3.0, 4.0], [4.0, 5.0, 1.0], [-1.3, 4.1, 9.0], [8.0, 110.3, 6.0]], requires_grad=True)\n";
    if(args.size() != 0)
        script << "\ty, idx = torch." << name << "(x" << args << ")\n";
    else
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

void make_clamp_files(std::string name, std::string args = "", std::string extension = ""){
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
    script << "\ta = torch.randint(-10, 10, (4, 5)).to(torch.float)\n";
    script << "\ta.requires_grad = True\n";
    script << "\ty = torch.clamp(a" << args << ")\n";
    script << "\tgrad = torch.rand_like(y)\n";
    script << "\ty.backward(gradient=grad)\n";
    script << "\tto_neurotensor(a.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_b << "')\n";
    script << "\tto_neurotensor(a.grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_d << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}

void make_imum_max_files(std::string name){
    std::string filename_a = "../tests/autograd_data/" + name + "_inputa.npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_inputb.npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_grad.npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_agrad.npy";
    std::string filename_e = "../tests/autograd_data/" + name + "_bgrad.npy";
    std::string filename_f = "../tests/autograd_data/" + name + "_output.npy";
    if(files_exist(filename_a, filename_b, filename_c, filename_d, filename_e, filename_f)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\ta = torch.tensor([1.0, 5.0, 2.0], requires_grad = True)\n";
    script << "\tb = torch.tensor([3.0, 2.0, 4.0], requires_grad = True)\n";
    script << "\ty = torch."<<name<<"(a, b)\n";
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

void min_max_autograd_test(){
    using namespace nt::literals;
    // flip test
    run_test("max (no dim) - Autograd test", [] {
        std::string name = "max";
        std::string extension = "";
        std::string args = "";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_min_max_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_output = nt::from_numpy(filename_d);
        auto o = nt::max(x);
        o.values.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.values.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.values.detach(), expected_output, nt::isclose(o.values.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
    });
    run_test("max (dim) - Autograd Test", [] {
        std::string name = "max";
        std::string extension = "_dim_one";
        std::string args = ", dim = 1";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_min_max_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_output = nt::from_numpy(filename_d);
        auto o = nt::max(x, dim = 1);
        o.values.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.values.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.values.detach(), expected_output, nt::isclose(o.values.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));

    });

    run_test("min (no dim) - Autograd test", [] {
        std::string name = "min";
        std::string extension = "";
        std::string args = "";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_min_max_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_output = nt::from_numpy(filename_d);
        auto o = nt::min(x);
        o.values.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.values.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.values.detach(), expected_output, nt::isclose(o.values.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
    });
    run_test("min (dim) - Autograd Test", [] {
        std::string name = "min";
        std::string extension = "_dim_one";
        std::string args = ", dim = 1";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_min_max_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_output = nt::from_numpy(filename_d);
        auto o = nt::min(x, dim = 1);
        o.values.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.values.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.values.detach(), expected_output, nt::isclose(o.values.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(x.grad(), expected_xgrad),
            "error: gradients for x do not match $ \n$ \n$ \n$",
            nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));

    });
    run_test("maximum - Autograd test", [] {
        std::string name = "maximum";
        std::string filename_a = "../tests/autograd_data/" + name + "_inputa.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_inputb.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_agrad.npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_bgrad.npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output.npy";
        make_imum_max_files(name);
        nt::TensorGrad a(nt::from_numpy(filename_a), true);
        nt::TensorGrad b(nt::from_numpy(filename_b), true);
        nt::Tensor grad = nt::from_numpy(filename_c);
        nt::Tensor expected_agrad = nt::from_numpy(filename_d);
        nt::Tensor expected_bgrad = nt::from_numpy(filename_e);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        auto o = nt::maximum(a, b);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(a.grad(), expected_agrad),
            "error: gradients for a do not match $ \n$ \n$ \n$",
            nt::noprintdtype, a.grad(), expected_agrad, nt::isclose(a.grad(), expected_agrad));
        nt::utils::throw_exception(
            nt::allclose(b.grad(), expected_bgrad),
            "error: gradients for b do not match $ \n$ \n$ \n$",
            nt::noprintdtype, b.grad(), expected_bgrad, nt::isclose(b.grad(), expected_bgrad));
    });

    run_test("minimum - Autograd test", [] {
        std::string name = "minimum";
        std::string filename_a = "../tests/autograd_data/" + name + "_inputa.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_inputb.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_agrad.npy";
        std::string filename_e = "../tests/autograd_data/" + name + "_bgrad.npy";
        std::string filename_f = "../tests/autograd_data/" + name + "_output.npy";
        make_imum_max_files(name);
        nt::TensorGrad a(nt::from_numpy(filename_a), true);
        nt::TensorGrad b(nt::from_numpy(filename_b), true);
        nt::Tensor grad = nt::from_numpy(filename_c);
        nt::Tensor expected_agrad = nt::from_numpy(filename_d);
        nt::Tensor expected_bgrad = nt::from_numpy(filename_e);
        nt::Tensor expected_output = nt::from_numpy(filename_f);
        auto o = nt::minimum(a, b);
        o.backward(grad);
        nt::utils::throw_exception(
            nt::allclose(o.detach(), expected_output),
            "Error: outputs do not match $ \n$ \n$ \n$",
            nt::noprintdtype, o.detach(), expected_output, nt::isclose(o.detach(), expected_output));
        nt::utils::throw_exception(
            nt::allclose(a.grad(), expected_agrad),
            "error: gradients for a do not match $ \n$ \n$ \n$",
            nt::noprintdtype, a.grad(), expected_agrad, nt::isclose(a.grad(), expected_agrad));
        nt::utils::throw_exception(
            nt::allclose(b.grad(), expected_bgrad),
            "error: gradients for b do not match $ \n$ \n$ \n$",
            nt::noprintdtype, b.grad(), expected_bgrad, nt::isclose(b.grad(), expected_bgrad));
    });

    run_test("clamp - Autograd test", [] {
        std::string name = "clamp";
        std::string extension = "";
        std::string args = ", min = -3, max = 3";
        std::string filename_a = "../tests/autograd_data/" + name + "_input" + extension + ".npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_grad" + extension + ".npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_xgrad" + extension + ".npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_output" + extension + ".npy";
        make_clamp_files(name, args, extension);
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor grad = nt::from_numpy(filename_b);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_c);
        nt::Tensor expected_output = nt::from_numpy(filename_d);
        auto o = nt::clamp(x, min = -3, max = 3);
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


    //run_test("clamp", [] {
    //    nt::Tensor a = nt::rand(1, 10, {5, 3, 8, 4, 2}, nt::DType::int32);
    //    nt::Tensor o = nt::clamp(a, min = 2, max = 8);
    //    nt::Scalar check = nt::max(o).values.toScalar();
    //    nt::utils::throw_exception(check.to<int64_t>() <= 8, "Error, max after clamp was $", check);
    //    check = nt::min(o).values.toScalar();
    //    nt::utils::throw_exception(check.to<int64_t>() <= 2, "Error, max after clamp was $", check);

    //});
    ////all the above tests are implemented to make sure they simply have working calls
    ////below argmin and argmax are more throughly tested because
    ////min and max functions rely on argmin and argmax's underlying functions to implement their actual logic
    //run_test("argmin", []{
    //    nt::Tensor a({4, 4}, nt::DType::Float32);
    //    a << 0.1139,  0.2254, -0.1381,  0.3687,
    //         1.0100, -1.1975, -0.0102, -0.4732,
    //        -0.9240,  0.1207, -0.7506, -1.0213,
    //         1.7809, -1.2960,  0.9384,  0.1438;
    //    nt::Scalar out_a = nt::argmin(a).toScalar();
    //    nt::utils::throw_exception(out_a.to<int64_t>() == 13, "Error, expected 13 got $", out_a);
    //    nt::Tensor out_b = nt::argmin(a, dim = 1);
    //    nt::Tensor check_b({4}, nt::DType::int64);

    //    check_b << 2,  1,  3,  1;
    //    nt::utils::throw_exception(nt::all(check_b == out_b), "Error, got invalid elements for argmin");
    //    nt::Tensor out_c = nt::argmin(a, dim = 1, keepdim=true);
    //    nt::utils::throw_exception(nt::all(out_c == check_b.view(4, 1)), "Error, got invalid elements for argmin");
    //});
    //run_test("argmax", []{
    //    nt::Tensor a({4, 4}, nt::DType::Float32);
    //    a << 1.3398,  0.2663, -0.2686,  0.2450,
    //        -0.7401, -0.8805, -0.3402, -1.1936,
    //         0.4907, -1.3948, -1.0691, -0.3132,
    //        -1.6092,  0.5419, -0.2993,  0.3195;
    //    nt::Scalar out_a = nt::argmax(a).toScalar();
    //    nt::utils::throw_exception(out_a.to<int64_t>() == 0, "Error, expected 13 got $", out_a);
    //    nt::Tensor out_b = nt::argmax(a, dim = 1);
    //    nt::Tensor check_b({4}, nt::DType::int64);
    //    check_b << 0,  2,  0,  1;
    //    nt::utils::throw_exception(nt::all(check_b == out_b), "Error, got invalid elements for argmin");
    //    nt::Tensor out_c = nt::argmax(a, dim = 1, keepdim=true);
    //    nt::utils::throw_exception(nt::all(out_c == check_b.view(4, 1)), "Error, got invalid elements for argmin");
    //});


}

