#ifndef NT_INTRUSIVE_VARIABLE_HPP__
#define NT_INTRUSIVE_VARIABLE_HPP__

#include "intrusive_ptr.hpp"

namespace nt{


template <typename T> class intrusive_variable : public intrusive_ptr_target {
    T var;

  public:
    using type = T;
    intrusive_variable(T val) : var(val) {}

    template <typename... Args>
    intrusive_variable(Args &&...args) : var(std::forward<Args>(args)...) {}

    T &get() { return var; }
    const T &get() const { return var; }
};

}

#endif
