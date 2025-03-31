#ifndef _NT_UTILS_OPTIONAL_EXCEPT_H_
#define _NT_UTILS_OPTIONAL_EXCEPT_H_


#include <stdexcept>

namespace nt{
namespace utils{


class bad_optional_access : public std::exception {
public:
    const char* what() const noexcept override {
        return "bad_optional_access";
    }
};



}} //nt::utils::

#endif // _NT_UTILS_OPTIONAL_EXCEPT_H_ 
