#ifndef NT_NN_BACKWARD_FUNC_H__
#define NT_NN_BACKWARD_FUNC_H__
namespace nt::grad::utility {

class backward_func;
class view_backward_func;
class self_mod_backward_func;

}
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../utils/api_macro.h"
#include "AutoGrad.h"
#include "forward_Grad.h"
#include <functional>
#include <string>

namespace nt::grad::utility {

class NEUROTENSOR_API backward_func : public intrusive_ptr_target {
    std::string name;
  public:
    using func_type = std::function<void(
        const Tensor &,
        std::vector<intrusive_ptr<TensorGrad>>&)>; // gradient, then the parents
  protected:
    func_type Func;


  public:

    backward_func()
        :Func(nullptr), name("NoneBackward") {}
    backward_func(std::string name_)
        :Func(nullptr), name(name_ + "Backward") {}
    backward_func(func_type func)
        :Func(func), name("NoneBackward") {}
    backward_func(func_type func, std::string name_)
        :Func(func), name(name_ + "Backward") {name[0] = std::toupper(name[0]);}

    inline virtual void set(func_type func) {
        Func = func;
    }
    
    inline virtual void set(std::nullptr_t) {
        Func = nullptr;
    }
    inline void set_name(std::string _name) noexcept {
        name = _name + "Backward";
        name[0] = std::toupper(name[0]);
    }
    inline const std::string &get_name() const noexcept { return name; }
    void run(const Tensor& grad, std::vector<intrusive_ptr<TensorGrad>>& v);
    void run(const Tensor& grad, const std::vector<weak_intrusive_ptr<GraphNode>>& weak_parents);
    inline bool is_valid() const noexcept {return Func != nullptr;}
    inline virtual bool is_view_change() const {return false;}
    inline virtual bool is_self_mod() const {return false;}

};

// 
class NEUROTENSOR_API self_mod_backward_func : public backward_func{
    public:
        self_mod_backward_func()
        :backward_func() {};
        self_mod_backward_func(std::string name_)
        :backward_func(std::move(name_)) {};
        self_mod_backward_func(backward_func::func_type func)
        :backward_func(func) {};
        
        inline bool is_self_mod() const override {return true;}

};

// constructor makes sure that there is not a function in view_backward_func
class NEUROTENSOR_API view_backward_func : public backward_func{
    public:
        view_backward_func()
        :backward_func() {this->Func = nullptr;}
        view_backward_func(std::string name_)
        :backward_func(std::move(name_)) {this->Func = nullptr;}
        view_backward_func(backward_func::func_type func)
        :backward_func() {
            utils::throw_exception(func == nullptr, "Error, tried to set view backward function gradient function");
            this->Func = nullptr;
        }
        view_backward_func(func_type func, std::string name_)
        :backward_func(std::move(name_)) {
            utils::throw_exception(func == nullptr, "Error, tried to set view backward function gradient function");
            this->Func = nullptr;
        }

        inline void set(backward_func::func_type func) override {
            utils::throw_exception(func == nullptr, "Error, tried to set view backward function gradient function");
            Func = nullptr;
        }
        
        inline void set(std::nullptr_t) override{
            Func = nullptr;
        }
        
        inline bool is_view_change() const override {return true;}

};

} // namespace nt::nn::utility

#endif
