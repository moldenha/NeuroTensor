#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>

/*
  
So, PyTorch only offers Unfold and Fold (this frameworks version of Unfold2D and Fold2D)
However, PyTorch does offer Conv1D, Conv2D, and Conv3D. NeuroTensor uses an unfold of dimension N and a matmult
If the conv autograd tests work, then it can be assumed that Unfold1D, Unfold2D, and Unfold3D work as expected
Plus, the following prove that Unfold2D and Fold2D work as expected

So for UnfoldND, since UnfoldND matches Unfold1D, Unfold2D, and Unfold3D in test mode -> (when unfoldnd is exclisively called for all dims)
 it is reasonable to assume that UnfoldND is properly generalized for all dimensions


For FoldND:
    - The following proves that Fold2D works,
    - Plus NeuroTensor uses Fold and Unfold as inverses of each other
    - If FoldND properly generalizes to Fold2D and it matches Fold1D and Fold3D
    - Then it can be generalized that Fold1D and Fold3D work along with FoldND
*/


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

void make_col_im_test_torch(std::string name, std::string shape, std::string args = "", std::string bounds="(-1, 1)"){
    std::string filename_a = "../tests/autograd_data/" + name +"_input.npy";
    std::string filename_b = "../tests/autograd_data/" + name + "_output.npy";
    std::string filename_c = "../tests/autograd_data/" + name + "_grad.npy";
    std::string filename_d = "../tests/autograd_data/" + name + "_input_grad.npy";
    if(files_exist(filename_a, filename_b,
                   filename_c, filename_d)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n"; 
    make_random_between(script, "x", shape, bounds);
    script << "\ty = torch.nn.functional." << name << "(x, " << args << ")\n";
    script << "\tgrad = torch.rand_like(y, requires_grad=False)\n";
    script << "\ty.backward(grad)\n";
    script << "\tto_neurotensor(x.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(y.detach(), '" << filename_b << "')\n";
    script << "\tto_neurotensor(grad, '" << filename_c << "')\n";
    script << "\tto_neurotensor(x.grad, '" << filename_d << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}


void col_im_test_autograd(){
    using namespace nt::literals;
    run_test("Unfold - autograd test", []{
        std::string name = "unfold";
        std::string filename_a = "../tests/autograd_data/" + name +"_input.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_output.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_input_grad.npy";
        make_col_im_test_torch(name, "1, 3, 8, 8", "kernel_size=(3,2), padding=1, stride=(2,2), dilation=(1,2)", "(0, 10)");
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor expected_output = nt::from_numpy(filename_b);
        nt::Tensor grad = nt::from_numpy(filename_c);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_d);
        nt::TensorGrad y = nt::unfold2d(x, {3, 2}, padding=1, stride = {2,2}, dilation = {1, 2});
        y.backward(grad);
        nt::utils::throw_exception(nt::allclose(y, expected_output), 
                               "Error, output from unfold to match the pytorch output $ \n$ \n$ \n$",
                               nt::noprintdtype, y.detach(), expected_output, nt::isclose(y, expected_output));
        nt::utils::throw_exception(nt::allclose(x.grad(), expected_xgrad), 
                               "Error, expected gradients to match $ \n$ \n$ \n$",
                               nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
    });
    run_test("Fold - autograd test", []{
        std::string name = "fold";
        std::string filename_a = "../tests/autograd_data/" + name +"_input.npy";
        std::string filename_b = "../tests/autograd_data/" + name + "_output.npy";
        std::string filename_c = "../tests/autograd_data/" + name + "_grad.npy";
        std::string filename_d = "../tests/autograd_data/" + name + "_input_grad.npy";
        make_col_im_test_torch(name, "3, 18, 25", "kernel_size=(3,2), output_size = 5, padding=1, dilation=(1,2)", "(0, 10)");
        nt::TensorGrad x(nt::from_numpy(filename_a), true);
        nt::Tensor expected_output = nt::from_numpy(filename_b);
        nt::Tensor grad = nt::from_numpy(filename_c);
        nt::Tensor expected_xgrad = nt::from_numpy(filename_d);
        nt::TensorGrad y = nt::fold2d(x, kernel_size = {3, 2}, output_size = 5, padding = 1, dilation = {1,2});
        y.backward(grad);
        nt::utils::throw_exception(nt::allclose(y, expected_output), 
                               "Error, output from fold to match the pytorch output $ \n$ \n$ \n$",
                               nt::noprintdtype, y.detach(), expected_output, nt::isclose(y, expected_output));
        nt::utils::throw_exception(nt::allclose(x.grad(), expected_xgrad), 
                               "Error, expected gradients to match $ \n$ \n$ \n$",
                               nt::noprintdtype, x.grad(), expected_xgrad, nt::isclose(x.grad(), expected_xgrad));
    });

    run_test("UnfoldND -> Unfold1D Autograd Test", []{
        nt::TensorGrad x1(nt::randn({2, 3, 10}), true);
        nt::TensorGrad x2(x1.detach().clone(), true);
        auto y = nt::unfold1d(x1, kernel_size=3, padding=1, stride=2, dilation=2);
        auto y2 = nt::functional::unfoldnd(x2, 1, /*kernel_size = */ 3,
                                           /*dilation = */ 2, 
                                           /*padding = */ 1, 
                                           /*stride = */ 2,
                                           /*transpose_out = */ true,
                                           /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y.detach(), y2.detach()), 
                            "Error outputs from UnfoldND and Unfold1D do not match");
        nt::Tensor grad1 = nt::randn(y.shape(), y.dtype());
        nt::Tensor grad2 = grad1.clone();
        y.backward(grad1);
        y2.backward(grad2);

        nt::utils::throw_exception(nt::allclose(x1.grad(), x2.grad()),
                               "Error, gradients for UnfoldND and Unfold1D do not match"
                                "$ \n$ \n$", nt::noprintdtype, x1.grad(), x2.grad());
    });
    
    run_test("UnfoldND -> Unfold2D Autograd Test", []{
        nt::TensorGrad x1(nt::rand(0, 10, {1, 3, 8, 8}), true);
        nt::TensorGrad x2(x1.detach().clone(), true);
        auto y = nt::unfold2d(x1, {3, 2}, padding = 1, stride = {2, 2}, dilation = {1, 2});
        auto y2 = nt::functional::unfoldnd(x2, 2, /*kernel_size = */ {3, 2},
                                           /*dilation = */ {1, 2}, 
                                           /*padding = */ 1, 
                                           /*stride = */ {2, 2},
                                           /*transpose_out = */ true,
                                           /*test_mode = */ true);        
        nt::utils::throw_exception(nt::allclose(y.detach(), y2.detach()), 
                            "Error outputs from UnfoldND and Unfold2D do not match");
        nt::Tensor grad1 = nt::randn(y.shape(), y.dtype());
        auto grad2 = grad1.clone();
        y.backward(grad1);
        y2.backward(grad2);

        nt::utils::throw_exception(nt::allclose(x1.grad(), x2.grad()),
                               "Error, gradients for UnfoldND and Unfold2D do not match");
    });

    run_test("UnfoldND -> Unfold3D Autograd Test", []{
        nt::TensorGrad x1(nt::rand(0, 10, {1, 2, 6, 6, 6}), true);
        nt::TensorGrad x2(x1.detach().clone(), true);
        auto y = nt::unfold3d(x1, {3, 3, 3}, padding = {1, 1, 1}, stride = {1, 2, 2}, dilation = 1);
        auto y2 = nt::functional::unfoldnd(x2, 3, /*kernel_size = */ {3, 3, 3},
                                           /*dilation = */ 1, 
                                           /*padding = */ {1, 1, 1}, 
                                           /*stride = */ {1, 2, 2},
                                           /*transpose_out = */ true,
                                           /*test_mode = */ true);    
        nt::utils::throw_exception(nt::allclose(y.detach(), y2.detach()), 
                            "Error outputs from UnfoldND and Unfold3D do not match");
        nt::Tensor grad1 = nt::randn(y.shape(), y.dtype());
        auto grad2 = grad1.clone();
        y.backward(grad1);
        y2.backward(grad2);

        nt::utils::throw_exception(nt::allclose(x1.grad(), x2.grad()),
                               "Error, gradients for UnfoldND and Unfold3D do not match");
    });

    run_test("FoldND -> Fold1D Autograd Test", []{
        nt::TensorGrad x1(nt::randn({3, 18, 3}), true);
        nt::TensorGrad x2(x1.detach().clone(), true);
        auto y = nt::fold1d(x1, kernel_size = 3, output_size = 5, padding = 1, dilation = 2);
        auto y2 = nt::functional::foldnd(x2, 1, /*output_size = */ 5, /*kernel_size = */3, /*dilation = */ 2, /*padding = */1,
                                         /*stride = */1, /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y.detach(), y2.detach()), 
                            "Error outputs from FoldND and Fold1D do not match");
        nt::Tensor grad1 = nt::randn(y.shape(), y.dtype());
        auto grad2 = grad1.clone();
        y.backward(grad1);
        y2.backward(grad2);

        nt::utils::throw_exception(nt::allclose(x1.grad(), x2.grad()),
                               "Error, gradients for FoldND and Fold1D do not match");
    });

    run_test("FoldND -> Fold2D Autograd Test", []{
        nt::TensorGrad x1(nt::randn({3, 18, 25}), true);
        nt::TensorGrad x2(x1.detach().clone(), true);
        auto y = nt::fold2d(x1, kernel_size = {3, 2}, output_size = 5, padding = 1, dilation = {1,2});
        auto y2 = nt::functional::foldnd(x2, 2, /*output_size = */ 5, /*kernel_size = */{3, 2}, /*dilation = */ {1, 2}, /*padding = */1,
                                         /*stride = */1, /*test_mode = */ true);
        nt::utils::throw_exception(nt::allclose(y.detach(), y2.detach()), 
                            "Error outputs from FoldND and Fold2D do not match");
        nt::Tensor grad1 = nt::randn(y.shape(), y.dtype());
        auto grad2 = grad1.clone();
        y.backward(grad1);
        y2.backward(grad2);

        nt::utils::throw_exception(nt::allclose(x1.grad(), x2.grad()),
                               "Error, gradients for FoldND and Fold2D do not match");
    });

 
    run_test("FoldND -> Fold3D Autograd Test", []{
        nt::TensorGrad x1(nt::randn({3, 24, 150}), true);
        nt::TensorGrad x2(x1.detach().clone(), true);
        auto y = nt::fold3d(x1, kernel_size = {3, 2, 2}, output_size = 5, padding = 1, dilation = {1, 1, 2});
        auto y2 = nt::functional::foldnd(x2, 3, /*output_size = */ 5, /*kernel_size = */{3, 2, 2}, /*dilation = */ {1, 1, 2}, /*padding = */1,
                                         /*stride = */1, /*test_mode = */ true);        
        nt::utils::throw_exception(nt::allclose(y.detach(), y2.detach()), 
                            "Error outputs from FoldND and Fold3D do not match");
        nt::Tensor grad1 = nt::randn(y.shape(), y.dtype());
        auto grad2 = grad1.clone();
        y.backward(grad1);
        y2.backward(grad2);

        nt::utils::throw_exception(nt::allclose(x1.grad(), x2.grad()),
                               "Error, gradients for FoldND and Fold3D do not match");
    });


}



// int main() {
//     conv_tests();
//     return 0;
// }
