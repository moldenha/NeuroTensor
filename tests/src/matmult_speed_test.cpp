#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include "test_macros.h"
#include <nt/dtype/ArrayVoid.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <chrono>

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

void make_matmult_time_files(){
    std::string filename_a = "../tests/autograd_data/matmult_speed_test_a.npy";
    std::string filename_b = "../tests/autograd_data/matmult_speed_test_b.npy";
    std::string filename_c = "../tests/autograd_data/matmult_speed_test_speed.npy";
    if(files_exist(filename_a, filename_b, filename_c)) return;
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import numpy as np\n";
    script << "import time\n";
    make_save_function(script);
    script << "def time_dot():\n";
    script << "\ta = np.random.rand(1000, 1000)\n";
    script << "\tb = np.random.rand(1000, 1000)\n";
    script << "\tstart = time.time()\n";
    script << "\tout = np.dot(a, b)\n";
    script << "\tend = time.time()\n";
    script << "\treturn (end - start), a, b";
    script << "\n\n";
    script << "def func_1():\n";
    script << "\ttime_dot()\n";
    script << "\telapsed, a, b = time_dot()\n";
    script << "\tnp.save('"<<filename_a<<"', a)\n";
    script << "\tnp.save('"<<filename_b<<"', b)\n";
    script << "\tnp.save('"<<filename_c<<"', np.array([elapsed]))\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
}





void matmult_speed_test(){
    using namespace nt::literals;
    run_test("Matmult Time test", []{
        make_matmult_time_files();
        nt::Tensor a = nt::from_numpy("../tests/autograd_data/matmult_speed_test_a.npy");
        nt::Tensor b = nt::from_numpy("../tests/autograd_data/matmult_speed_test_b.npy");
        nt::Tensor elapsed = nt::from_numpy("../tests/autograd_data/matmult_speed_test_speed.npy");
         
        nt::Tensor out = nt::functional::matmult(a, b);
        auto start = std::chrono::high_resolution_clock::now();
        nt::Tensor out2 = nt::functional::matmult(a, b);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        auto microseconds = duration.count();
        double seconds = static_cast<double>(microseconds) / 1000000.0; 


        std::cout << "numpy elapsed (seconds): " << elapsed.item<double>()<<std::endl;
        std::cout << "neuro tensor elapsed (seconds): "<< seconds << std::endl;

    });


}

