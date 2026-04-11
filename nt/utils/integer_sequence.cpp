// this is a test file for the integer sequence

#include "integer_sequence.hpp"
#include <iostream>
#include <type_traits>

template<typename Tv, typename T, T... integers>
std::vector<std::reference_wrapper<Tv>> make_vec(std::vector<Tv>& vec, nt::utils::integer_sequence<T, integers...>){
    return std::vector<std::reference_wrapper<Tv>>{
        std::ref(vec[integers])...
    };
}

int main(){
    using float_sequence = nt::utils::is_same_index_sequence<float, int, float, int, double, float, int, float>;
    using int_sequence = nt::utils::is_same_index_sequence<int, int, float, int, double, float, int, float>;
    using fi_sequence = nt::utils::integer_sequence_concat<float_sequence, int_sequence>::type;
    float_sequence sequence_f = float_sequence{};
    int_sequence sequence_i = int_sequence{};
    fi_sequence sequence_fi = fi_sequence{};
    std::cout << sequence_f << std::endl;
    std::cout << sequence_i << std::endl;
    std::cout << sequence_fi << std::endl;

    std::vector<int> test_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<std::reference_wrapper<int>> ref_vec = make_vec(test_vec, fi_sequence{});
    for(auto& ref : ref_vec){
        std::cout << ref.get() << ' ';
        ref.get() += 100;
    }
    std::cout << std::endl;
    for(const auto& val : test_vec)
        std::cout << val << ' ';
    std::cout << std::endl;
    std::cout << std::boolalpha << nt::utils::is_index_sequence_in(1, fi_sequence{})
                        << ',' << nt::utils::is_index_sequence_in(4, fi_sequence{})
                        << ',' << nt::utils::is_index_sequence_in(3, fi_sequence{})
                        << std::noboolalpha << std::endl;
    

    return 0;
}
