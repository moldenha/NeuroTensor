#ifndef NT_TEST_MACROS_H__
#define NT_TEST_MACROS_H__ 
//this is a header file to standardize testing
#include <nt/utils/macros.h>
#include <iostream>
#include <vector>


#define NT_TEST_FUNCTION(name, ...)\
    auto test_output = name(__VA_ARGS__);\
    std::cout << test_output << std::endl;

#define NT_TEST_ARGS(...) __VA_ARGS__


#define NT_TEST_EQUAL(name, ARGS_1, ARGS_2)\
    auto output_a = name(ARGS_1);\
    auto output_b = name(ARGS_2);\
    std::cout << std::boolalpha << nt::equal(output_a, output_b) << std::endl;


// Primary template
template<typename T>
struct DTypeEnumToVector {
    static std::vector<nt::DType> value;
};

// Specialization for nt::DTypeEnum
template<nt::DType... Ts>
struct DTypeEnumToVector<nt::DTypeEnum<Ts...>> {
    static std::vector<nt::DType> value;
};

// Definition outside
template<nt::DType... Ts>
std::vector<nt::DType> DTypeEnumToVector<nt::DTypeEnum<Ts...>>::value = { Ts... };

// Using DTypeEnum to vector because the following types automatically detect compatible DTypes
// and compatible types on the users computer (also faster from a development standpoint)
// The following can be used like such:
// for(const nt::DType& dt : NumberTypes){
//    // and then test unique for example for all the dtypes with a simple loop
// }
//
// The reason for this is that it is obviously important that the functions work for all the intended types
// This is an easy way to automate that checking
static std::vector<nt::DType> NumberTypes = DTypeEnumToVector<nt::NumberTypesL>::value;
static std::vector<nt::DType> FloatingTypes = DTypeEnumToVector<nt::FloatingTypesL>::value;
static std::vector<nt::DType> ComplexTypes = DTypeEnumToVector<nt::ComplexTypesL>::value;
static std::vector<nt::DType> IntegerTypes = DTypeEnumToVector<nt::IntegerTypesL>::value;
static std::vector<nt::DType> SignedTypes = DTypeEnumToVector<nt::SignedTypesL>::value;
static std::vector<nt::DType> UnsignedTypes = DTypeEnumToVector<nt::UnsignedTypesL>::value;
static std::vector<nt::DType> AllTypes = DTypeEnumToVector<nt::AllTypesL>::value;
static std::vector<nt::DType> AllTypesNBool = DTypeEnumToVector<nt::AllTypesNBoolL>::value;
static std::vector<nt::DType> RealNumberTypes = DTypeEnumToVector<nt::RealNumberTypesL>::value;


inline std::vector<nt::DType> combine(const std::vector<nt::DType>& a, const std::vector<nt::DType>& b){
    std::vector<nt::DType> n(a.size() + b.size());
    std::copy(b.cbegin(), b.cend(), std::copy(a.cbegin(), a.cend(), n.begin()));
    return std::move(n);
}


// std::cout << "\033[31mRed text\033[0m\n";
// std::cout << "\033[32mGreen text\033[0m\n";
// std::cout << "\033[1;34mBold Blue text\033[0m\n";
template<typename Func>
inline void run_test(const std::string& name, Func&& f) {
    try {
        f(); // Run the test
        if(nt::utils::getAllocatedMemory(nt::DeviceType::CPU) != 0){
            std::cout << "\033[1;31m[✗]\033[0m " << name << " worked but left with "<<nt::utils::getAllocatedMemory(nt::DeviceType::CPU)<<" bytes of memory allocated in CPU \n";
            return;
            
        }
        if(nt::utils::getAllocatedMemory(nt::DeviceType::META) != 0){
            std::cout << "\033[1;31m[✗]\033[0m " << name << " worked but left with "<<nt::utils::getAllocatedMemory(nt::DeviceType::META)<<" bytes of memory allocated in META \n";
            for(const auto& [key, val] : nt::utils::memory_details::getMetaRegistry().unsafe_get_cref()){
                auto [size, file, line] = val;
                std::cout << "\tMemory of "<<size<<" bytes was allocated in "<<file<<" on line" << line<<std::endl;
            }
            return;
            
        }

        std::cout << "\033[32m[✓]\033[0m " << name << " works \n";
    } catch (const std::exception& e) {
        std::cout << "\033[1;31m[✗]\033[0m " << name << " did not work — " << e.what() << "\n";
    } catch (...) {
        std::cout << "\033[1;31m[✗]\033[0m " << name << " did not work — unknown error\n";
    }
}


#endif
