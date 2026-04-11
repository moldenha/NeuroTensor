#include <memory>
#include <iostream>
#include <cmath>
#include "mtl_fuse_element_str_builder.h"

struct example{
    std::shared_ptr<int> i;
    example()
        :i(std::make_shared<int>(1))
    {}
    example(int a)
        :i(std::make_shared<int>(a))
    {}
};

example Add(const example& a, const example& b){
    std::cout << "Add: " <<a.i.use_count() << ' ' << b.i.use_count() << std::endl;
    return example(*a.i + *b.i);
}


example Relu(const example& a){
    std::cout << "Relu: " << a.i.use_count() << std::endl;
    return example(std::max(0, *a.i));
}


int main(){
    example a(10);
    example b(3);
    example c = Relu(Add(a, b));
    std::cout << nt::mtl::fuse::get_var_name(4, 56, nt::mtl::fuse::FuseVars::Num<3>()) << std::endl;
    
    auto fuse = nt::mtl::fuse::build_fuse_element(
                                /*inputting_variables = */2,
                                nt::mtl::fuse::FuseVars::Index<0>(),
                                " + ",
                                nt::mtl::fuse::FuseVars::Index<1>());

    // ReLU
    fuse = nt::mtl::fuse::build_fuse_element(fuse, 
                                    /*inputting_variables = */ 1,
                                    "max({type}(0), ", nt::mtl::fuse::FuseVars::Functional::LastResult,
                                    ") - ", nt::mtl::fuse::FuseVars::Index<0>());
    std::cout << nt::mtl::fuse::build_result(0, fuse.infos[0], fuse) << std::endl;
    std::cout << nt::mtl::fuse::build_result(1, fuse.infos[0], fuse) << std::endl;
    // // adding 3
    // this would simulate another tensor that is having fuse ops happening on it
    auto fuse_2 = nt::mtl::fuse::build_fuse_element(
                                                    /* inputting_variables = */ 1,
                                                    nt::mtl::fuse::FuseVars::Index<0>(), " + 3");
    fuse_2 = nt::mtl::fuse::build_fuse_element(fuse_2,
                                /*inputting_variables = */ 1,
                                nt::mtl::fuse::FuseVars::Index<0>(), " * ",
                                nt::mtl::fuse::FuseVars::Functional::LastResult);
    auto fuse_3 = nt::mtl::fuse::build_fuse_element(std::vector<std::pair<size_t,
                                    std::reference_wrapper<const nt::mtl::fuse::FuseElementHolders>>>({{1, std::cref(fuse)}, {2, std::cref(fuse_2)}}),
                                                /* inputting_variables =*/3, // fuse is variable (1) and fuse_2 is variable(2) and the new tensor is variable (0)
                                            nt::mtl::fuse::FuseVars::Index<0>(), " + (",
                                            nt::mtl::fuse::FuseVars::Index<1>(), " * ",
                                            nt::mtl::fuse::FuseVars::Index<2>(), ")");
    // std::cout << "built fuse 3" << std::endl; 
    // std::string total_calcs = nt::mtl::fuse::build_results(fuse_3);
    // std::cout << total_calcs << std::endl;
    
    // fuse_str = nt::mtl::fuse::build_fuse_element(fuse_str,
    //                                                 /*inputting_variables = */0,
    //                                                 /*total_previous_variables = */3,
    //                                                 nt::mtl::fuse::FuseVars::Functional::Type, "(3) + (", 
    //                                                 nt::mtl::fuse::FuseVars::Functional::LastResult, ")");

    // // exp function
    // fuse_str = nt::mtl::fuse::build_fuse_element(fuse_str,
    //                                                 /*inputting_variables = */1,
    //                                                 /*total_previous_variables = */3,
    //                                                 "exp(", nt::mtl::fuse::FuseVars::Functional::LastResult, 
    //                                                 ") * ", nt::mtl::fuse::FuseVars::Index<0>());

    
    // // building the final function
    std::string mtl_function = nt::mtl::fuse::build_fuse_element_function(fuse_3,
                                                            /*type = */ "float",
                                                            /*name = */ "fused_function");
    std::cout << mtl_function << std::endl;

    return 0;

}
