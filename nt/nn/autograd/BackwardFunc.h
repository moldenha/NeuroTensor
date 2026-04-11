#ifndef NT_NN_BACKWARD_FUNC_H__
#define NT_NN_BACKWARD_FUNC_H__
namespace nt::grad::utility {

enum class EdgeType{
    Read,
    Write,
    View,
    None
};

inline constexpr EdgeType KEdgeRead  = EdgeType::Read;
inline constexpr EdgeType KEdgeWrite = EdgeType::Write;
inline constexpr EdgeType KEdgeView  = EdgeType::View;
inline constexpr EdgeType KEdgeNone  = EdgeType::None;

class backward_func;
class view_backward_func;
class self_mod_backward_func;


}

#include <functional>
#include <vector>
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../forward_Grad.h"
#include <memory>
#include <string>

namespace nt::grad::utility {

using backward_func_type = std::function<void(
    const Tensor&,
    std::vector<intrusive_ptr<TensorGrad>>&
)>;

std::unique_ptr<backward_func> make_backward_func(EdgeType edge, backward_func_type func, std::string name);

}
#include "../../utils/api_macro.h"
// #include "AutoGrad.h"
#include "GraphNode.h"

namespace nt::grad::utility {


// Use std::unique_ptr<backward_func> 

// class NEUROTENSOR_API backward_func : public intrusive_ptr_target {

class NEUROTENSOR_API backward_func {
    std::string name;
  public:
    using func_type = backward_func_type; // gradient, then the parents
  protected:
    func_type Func;


  public:

    backward_func()
        :Func(nullptr), name("NoneBackward") {}
    backward_func(std::nullptr_t)
        :Func(nullptr), name("NoneBackward") {}
    backward_func(std::string name_)
        :Func(nullptr), name(name_ + "Backward") {}
    backward_func(func_type func)
        :Func(func), name("NoneBackward") {}
    backward_func(func_type func, std::string name_)
        :Func(func), name(name_ + "Backward") {name[0] = std::toupper(name[0]);}
    
    inline virtual const EdgeType& edge_type() const { return KEdgeRead;}

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
    void run(const Tensor& grad, const std::vector<std::pair<weak_intrusive_ptr<GraphNode>, uint64_t>>& weak_parents);
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
        self_mod_backward_func(backward_func::func_type func, std::string name)
        :backward_func(func, std::move(name))
        {}
        
        inline const EdgeType& edge_type() const override { return KEdgeWrite;}
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
        :backward_func() {}
        view_backward_func(func_type func, std::string name_)
        :backward_func(std::move(name_)) {}        
        
        inline const EdgeType& edge_type() const override { return KEdgeView;}
        inline bool is_view_change() const override {return true;}

};

inline std::unique_ptr<backward_func> make_backward_func(EdgeType edge, backward_func::func_type func, std::string name){
    switch(edge){
        case EdgeType::Read:
            return std::make_unique<backward_func>(std::move(func), std::move(name));
        case EdgeType::Write:
            return std::make_unique<self_mod_backward_func>(std::move(func), std::move(name));
        case EdgeType::View:
            return std::make_unique<view_backward_func>(std::move(func), std::move(name));
        case EdgeType::None:
            return std::make_unique<backward_func>();
    }
    return std::make_unique<backward_func>(nullptr);
}

} // namespace nt::grad::utility

#endif
