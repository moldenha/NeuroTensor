#ifndef NT_UTILS_OPTIONAL_EXCEPT_H__
#define NT_UTILS_OPTIONAL_EXCEPT_H__


#include <stdexcept>
#include "api_macro.h"

namespace nt{
namespace utils{


class NEUROTENSOR_API bad_optional_access : public std::exception {
public:
    inline const char* what() const noexcept override {
        return "bad_optional_access";
    }
};

class NEUROTENSOR_API bad_optional_size_access : public std::exception {
public:
    inline const char* what() const noexcept override {
        return "bad_optional_size_access";
    }
};


}} //nt::utils::

#endif // _NT_UTILS_OPTIONAL_EXCEPT_H_ 
