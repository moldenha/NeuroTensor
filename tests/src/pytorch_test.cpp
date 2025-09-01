#define NT_DEFINE_PARAMETER_ARGUMENTS 
#include <nt/nt.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>


inline std::ofstream& make_save_function(std::ofstream& script){
    script << "def to_neurotensor(tensor, name):\n";
    script << "\tt_n = tensor.numpy()\n";
    script << "\tnp.save(name, t_n)\n\n";
    return script;
}

void __pytorch_test__(){
    const std::string script_name = "session.py";
    std::ofstream script(script_name);
    script << "# Python session\n";
    script << "import torch\n";
    script << "import numpy as np\n";
    make_save_function(script);
    script << "def func_1():\n";
    script << "\tt = torch.rand(3, 4, 5)\n";
    script << "\tprint(t)\n";
    script << "\tto_neurotensor(t, 'tn_example.npy')\n";
    script << "\n\n";
    script << "func_1()\n";
    script.close();

    std::string cmd = "python3 "+script_name;
    std::system(cmd.c_str());
    nt::Tensor t = nt::from_numpy("tn_example.npy");
    std::cout << t << std::endl;
 
}

void pytorch_test(){
    // std::string line;
    // std::cout << "Python session CLI (type 'exit' to quit):\n";

    // while (true) {
    //     std::cout << ">>> ";
    //     std::getline(std::cin, line);
    //     if (line == "exit") break;

    //     std::ofstream append(script_name, std::ios::app);
    //     append << line << "\n";
    //     append.close();

    //     std::string cmd = "python3 " + script_name;
    //     std::system(cmd.c_str());
    // }
}
