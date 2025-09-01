#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>


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

void make_range_test_torch(){
    std::string filename_a = "../tests/autograd_data/range_input.npy";
    std::string filename_b = "../tests/autograd_data/range_output_a.npy";
    std::string filename_c = "../tests/autograd_data/range_output_b.npy";
    std::string filename_d = "../tests/autograd_data/range_output_c.npy";

    if(files_exist(filename_a, filename_b,
                   filename_c, filename_d)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n"; 
    script << "\tt = torch.arange(5*6*7*8).view(5, 6, 7, 8)\n";
    script << "\tt1 = t[:, 1 : 4, :, 3 :]\n";
    script << "\tt2 = t1[ 1 : 4, 1 :, 2 : 6]\n";
    script << "\tt3 = t2.transpose(-1, -2)\n";
    script << "\tt4 = t3[1 :, :, 1 : 3]\n";
    script << "\tto_neurotensor(t.detach(), '" << filename_a << "')\n";
    script << "\tto_neurotensor(t1.detach(), '" << filename_b << "')\n";
    script << "\tto_neurotensor(t2.detach(), '" << filename_c << "')\n";
    script << "\tto_neurotensor(t4.detach(), '" << filename_d << "')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());

}


void range_test(){
    using namespace nt::literals;
    run_test("Range test", []{
        std::string filename_a = "../tests/autograd_data/range_input.npy";
        std::string filename_b = "../tests/autograd_data/range_output_a.npy";
        std::string filename_c = "../tests/autograd_data/range_output_b.npy";
        std::string filename_d = "../tests/autograd_data/range_output_c.npy";
        make_range_test_torch();
        nt::Tensor t = nt::from_numpy(filename_a);
        nt::Tensor t1 = nt::from_numpy(filename_b);
        nt::Tensor t2 = nt::from_numpy(filename_c);
        nt::Tensor t3 = nt::from_numpy(filename_d);

        nt::Tensor x1 = t( nt::range, 1 < nt::range > 4, nt::range, 3 < nt::range);
        nt::Tensor x2 = x1(1 < nt::range > 4, 1 < nt::range > -1, 2 < nt::range > 6);
        nt::Tensor x3 = x2.transpose(-1, -2);
        nt::Tensor x4 = x3(1 < nt::range > -1, nt::range, 1 < nt::range > 3);
        nt::utils::throw_exception(
            nt::all(x1 == t1),
            "Error, output from first range check [:, 1 : 4, :, 3 : -1] does not match pytorch output"
            "$ \n$ \n$", nt::noprintdtype, x1, t1); // standard check
        nt::utils::throw_exception(
            nt::all(x2 == t2),
            "Error, output from second range check [ 1 : 4, 1 : -1, 2 : 6] does not match pytorch output"
            "$ \n$ \n$", nt::noprintdtype, x2, t2); // blocked data range check
        nt::utils::throw_exception(
            nt::all(x4 == t3),
            "Error, output from third range check [1 : -1, :, 1 : 3] does not match pytorch output"
            "$ \n$ \n$", nt::noprintdtype, x4, t3); // strided data range check
    });
}

