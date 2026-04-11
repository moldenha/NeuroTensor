#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "../TensorGrad.h"
#include "../functional_class.h"
#include "../../utils/macros.h"

namespace nt{
namespace functional{


//returns c + (a * b);
TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const TensorGrad& c, const TensorGrad& a, const TensorGrad& b){
    if(!c.track_grad() || !a.track_grad() || !b.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c.detach(), a, b);
        else if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c, a.detach(), b);
        else if(!b.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c, a, b.detach());
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c.detach(), a.detach(), b.detach()), c, a, b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of A
            //parents[2]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * b->tensor);
            parents[2]->accumulate_gradient(grad * a->tensor);

        }, a, b);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const Tensor& c, const TensorGrad& a, const TensorGrad& b){
    if(!a.track_grad() || !b.track_grad()){
        if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c, a.detach(), b);
        else if(!b.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c, a, b.detach());
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c, a.detach(), b.detach()), a, b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents, 
                intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of A
            //parents[1]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad * b->tensor);
            parents[1]->accumulate_gradient(grad * a->tensor);

        }, a, b);
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const Tensor& c, const Tensor& a, const TensorGrad& b){


    if(!b.track_grad()){ return TensorGrad(::nt::functional::fused_multiply_add(c, a, b.detach()), false); }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c, a, b.detach()), b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents, 
                intrusive_ptr<tensor_holder> a) {
            //parents[0]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad * a->tensor);

        }, a);
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const Tensor& c, const TensorGrad& a, const Tensor& b){
    

    if(!a.track_grad()) return TensorGrad(::nt::functional::fused_multiply_add(c, a.detach(), b), false);   
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c, a.detach(), b), a);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents, 
                intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad * b->tensor);

        }, b);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const TensorGrad& c, const Tensor& a, const TensorGrad& b){
    if(!c.track_grad() || !b.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c.detach(), a, b);
        else if(!b.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c, a, b.detach());
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c.detach(), a, b.detach()), c, b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents, 
                intrusive_ptr<tensor_holder> a) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * a->tensor);

        }, a);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const TensorGrad& c, const Tensor& a, const Tensor& b){
    if(!c.track_grad()) return TensorGrad( ::nt::functional::fused_multiply_add(c.detach(), a, b), false); 
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c.detach(), a, b), c);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of C
            parents[0]->accumulate_gradient(grad);

        });
    return std::move(result);

}

TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const TensorGrad& c, const TensorGrad& a, const Tensor& b){
    if(!c.track_grad() || !a.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c.detach(), a, b);
        else if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c, a.detach(), b);
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c.detach(), a.detach(), b), c, a);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * b->tensor);

        }, b);
    return std::move(result);
 
}

TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const TensorGrad& c, const TensorGrad& a, Scalar b){
    if(!c.track_grad() || !a.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c.detach(), a, b);
        else if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_add(c, a.detach(), b);
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c.detach(), a.detach(), b), c, a);
    result.create_read_backward_function(
        [b](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * b);

        });
    return std::move(result);
 
}
TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const TensorGrad& c, const Tensor& a, Scalar b){
    if(!c.track_grad()) return TensorGrad( ::nt::functional::fused_multiply_add(c.detach(), a, b), false);

    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c.detach(), a, b), c);
    result.create_read_backward_function(
        [b](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of C
            parents[0]->accumulate_gradient(grad);

        });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fused_multiply_add(const Tensor& c, const TensorGrad& a, Scalar b){
    if(!a.track_grad()) return TensorGrad(::nt::functional::fused_multiply_add(c, a.detach(), b), false);  

    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_add(c, a.detach(), b), a);
    result.create_read_backward_function(
        [b](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad * b);

        });
    return std::move(result);

}
//returns c += (a * b);


template<typename T1, typename T2>
inline void handle_null_tensors(const T1& t1, const T2& t2, const Scalar& s, const char* func_name = __NT_FUNCTION_NAME__){
    utils::throw_exception(!t1.is_null() && !t2.is_null(),
                           "Unable to perform operation $ on null tensors", func_name);
}


template<typename T1, typename T2, typename T3>
inline void handle_null_tensors(const T1& t1, const T2& t2, const T3& t3, const char* func_name = __NT_FUNCTION_NAME__){
    utils::throw_exception(!t1.is_null() && !t2.is_null() && !t3.is_null(),
                           "Unable to perform operation $ on null tensors", func_name);
}

#ifdef _MSC_VER
#define NT_INSIDE_FUNC_NAME __func__
#else
#define NT_INSIDE_FUNC_NAME __NT_FUNCTION_NAME__
#endif

inline const Tensor& resolve_tensor(const Tensor& t){return t;}
inline const Tensor& resolve_tensor(const TensorGrad& t){return t.detach();}

inline bool get_do_track_grad(const Tensor& t){return true;} // so that it isn't perpetually having the same function called
inline bool get_do_track_grad(const TensorGrad& t){return t.track_grad();}


TensorGrad& TensorGrad_Functional_Class::fused_multiply_add_(TensorGrad& c, const TensorGrad& a, const TensorGrad& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    if(!get_do_track_grad(a)){return TensorGrad_Functional_Class::fused_multiply_add_(c, resolve_tensor(a), b);}
    if(!get_do_track_grad(b)){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, resolve_tensor(b));}
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    intrusive_ptr<tensor_holder> _a = make_intrusive<tensor_holder>(resolve_tensor(a));
    intrusive_ptr<tensor_holder> _b = make_intrusive<tensor_holder>(resolve_tensor(b));
    c.create_write_node([_a, _b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * _b->tensor);
        parents[2]->accumulate_gradient(grad * _a->tensor);
    }, __func__, a, b);

    return c;
}

TensorGrad& TensorGrad_Functional_Class::fused_multiply_add_(TensorGrad& c, const Tensor& a, const TensorGrad& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    if(!get_do_track_grad(b)){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, resolve_tensor(b));}
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    intrusive_ptr<tensor_holder> _a = make_intrusive<tensor_holder>(resolve_tensor(a));
    c.create_write_node([_a](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * _a->tensor);
    }, __func__, b);

    return c;

}
TensorGrad& TensorGrad_Functional_Class::fused_multiply_add_(TensorGrad& c, const TensorGrad& a, const Tensor& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    if(!get_do_track_grad(a)){return TensorGrad_Functional_Class::fused_multiply_add_(c, a, resolve_tensor(a));}
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    intrusive_ptr<tensor_holder> _b = make_intrusive<tensor_holder>(resolve_tensor(b));
    c.create_write_node([_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * _b->tensor);
    }, __func__, a);
    return c;
}

TensorGrad& TensorGrad_Functional_Class::fused_multiply_add_(TensorGrad& c, const Tensor& a, const Tensor& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_add_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    c.create_write_node([](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
    }, __func__);
    return c;
}


TensorGrad& TensorGrad_Functional_Class::fused_multiply_add_(TensorGrad& c, const TensorGrad& a, Scalar b){
    if(!a.track_grad()){
        return TensorGrad_Functional_Class::fused_multiply_add_(c, a.detach(), b);
    }if(!c.track_grad()){
        ::nt::functional::fused_multiply_add_(c.detach(), a.detach(), b);
        return c;
    }
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_add_(c.detach(), a.detach(), b);
    c.create_write_node([b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * b);
    }, "FusedMultiplyAdd_", a);
    return c;
}

TensorGrad& TensorGrad_Functional_Class::fused_multiply_add_(TensorGrad& c, const Tensor& a, Scalar b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_add_(c.detach(), a, b);
        return c;
    }
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_add_(c.detach(), a, b);
    c.create_write_node([](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
    }, "FusedMultiplyAdd_");
    return c;
}
 

//returns c - (a * b);
TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, const TensorGrad& b){
    if(!c.track_grad() || !a.track_grad() || !b.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c.detach(), a, b);
        else if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c, a.detach(), b);
        else if(!b.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b.detach());
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c.detach(), a.detach(), b.detach()), c, a, b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of A
            //parents[2]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * (-b->tensor));
            parents[2]->accumulate_gradient(grad * (-a->tensor));

        }, a, b);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const Tensor& c, const TensorGrad& a, const TensorGrad& b){
    if(!a.track_grad() || !b.track_grad()){
        if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c, a.detach(), b);
        else if(!b.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b.detach());
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c, a.detach(), b.detach()), a, b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> a, intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of A
            //parents[1]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad * (-b->tensor));
            parents[1]->accumulate_gradient(grad * (-a->tensor));

        }, a, b);
    return std::move(result);

}
TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const Tensor& c, const Tensor& a, const TensorGrad& b){

    if(!b.track_grad()) return TensorGrad(::nt::functional::fused_multiply_subtract(c, a, b.detach()), false); 
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c, a, b.detach()), b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> a) {
            //parents[0]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad * (-a->tensor));

        }, a);
    return std::move(result);
}
TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const Tensor& c, const TensorGrad& a, const Tensor& b){
    if(!a.track_grad()) return TensorGrad(::nt::functional::fused_multiply_subtract(c, a.detach(), b), false); 
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c, a.detach(), b), a);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad * (-b->tensor));

        }, b);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const TensorGrad& c, const Tensor& a, const TensorGrad& b){
    if(!c.track_grad() || !b.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c.detach(), a, b);
        else if(!b.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c, a, b.detach());
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c.detach(), a, b.detach()), c, b);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> a) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of B
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * (-a->tensor));

        }, a);
    return std::move(result);
}


TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const TensorGrad& c, const Tensor& a, const Tensor& b){
    if(!c.track_grad()) return TensorGrad(::nt::functional::fused_multiply_subtract(c.detach(), a, b), false); 
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c.detach(), a, b), c);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of C
            parents[0]->accumulate_gradient(grad);

        });
    return std::move(result);

}

TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, const Tensor& b){
    if(!c.track_grad() || !a.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c.detach(), a, b);
        else if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c, a.detach(), b);
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c.detach(), a.detach(), b), c, a);
    result.create_read_backward_function(
        [](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents,
                intrusive_ptr<tensor_holder> b) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * (-b->tensor));

        }, b);
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const TensorGrad& c, const TensorGrad& a, Scalar b){
    if(!c.track_grad() || !a.track_grad()){
        if(!c.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c.detach(), a, b);
        else if(!a.track_grad()) return TensorGrad_Functional_Class::fused_multiply_subtract(c, a.detach(), b);
    }
    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c.detach(), a.detach(), b), c, a);
    result.create_read_backward_function(
        [b](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of C
            //parents[1]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad);
            parents[1]->accumulate_gradient(grad * -b);

        });
    return std::move(result);
 
}
TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const TensorGrad& c, const Tensor& a, Scalar b){
    if(!c.track_grad()) return TensorGrad(::nt::functional::fused_multiply_subtract(c.detach(), a, b), false); 

    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c.detach(), a, b), c);
    result.create_read_backward_function(
        [b](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of C
            parents[0]->accumulate_gradient(grad);

        });
    return std::move(result);
}

TensorGrad TensorGrad_Functional_Class::fused_multiply_subtract(const Tensor& c, const TensorGrad& a, Scalar b){
    if(!a.track_grad()) return TensorGrad(::nt::functional::fused_multiply_subtract(c, a.detach(), b), false); 

    TensorGrad result = TensorGrad::create_read_node(::nt::functional::fused_multiply_subtract(c, a.detach(), b), a);
    result.create_read_backward_function(
        [b](const Tensor &grad,
                  std::vector<intrusive_ptr<TensorGrad>> &parents) {
            //parents[0]->grad->tensor is the gradient of A
            parents[0]->accumulate_gradient(grad * -b);

        });
    return std::move(result);

}


TensorGrad& TensorGrad_Functional_Class::fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, const TensorGrad& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    if(!get_do_track_grad(a)){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, resolve_tensor(a), b);}
    if(!get_do_track_grad(b)){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, resolve_tensor(b));}
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    intrusive_ptr<tensor_holder> _a = make_intrusive<tensor_holder>(resolve_tensor(a));
    intrusive_ptr<tensor_holder> _b = make_intrusive<tensor_holder>(resolve_tensor(b));
    c.create_write_node([_a, _b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * (-_b->tensor));
        parents[2]->accumulate_gradient(grad * (-_a->tensor));
    }, __func__, a, b);

    return c;
}



TensorGrad& TensorGrad_Functional_Class::fused_multiply_subtract_(TensorGrad& c, const Tensor& a, const TensorGrad& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    if(!get_do_track_grad(b)){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, resolve_tensor(b));}
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    intrusive_ptr<tensor_holder> _a = make_intrusive<tensor_holder>(resolve_tensor(a));
    c.create_write_node([_a](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * (-_a->tensor));
    }, __func__, b);

    return c;

}
TensorGrad& TensorGrad_Functional_Class::fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, const Tensor& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    if(!get_do_track_grad(a)){return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a, resolve_tensor(a));}
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    intrusive_ptr<tensor_holder> _b = make_intrusive<tensor_holder>(resolve_tensor(b));
    c.create_write_node([_b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * (-_b->tensor));
    }, __func__, a);
    return c;
}

TensorGrad& TensorGrad_Functional_Class::fused_multiply_subtract_(TensorGrad& c, const Tensor& a, const Tensor& b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
        return c;
    }
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_subtract_(c.detach(), resolve_tensor(a), resolve_tensor(b));
    c.create_write_node([](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
    }, __func__);
    return c;
}


TensorGrad& TensorGrad_Functional_Class::fused_multiply_subtract_(TensorGrad& c, const TensorGrad& a, Scalar b){
    if(!a.track_grad()){
        return TensorGrad_Functional_Class::fused_multiply_subtract_(c, a.detach(), b);
    }if(!c.track_grad()){
        ::nt::functional::fused_multiply_subtract_(c.detach(), a.detach(), b);
        return c;
    }
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_subtract_(c.detach(), a.detach(), b);
    c.create_write_node([b](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
        parents[1]->accumulate_gradient(grad * (-b));
    }, "FusedMultiplySubtract_", a);
    return c;
}

TensorGrad& TensorGrad_Functional_Class::fused_multiply_subtract_(TensorGrad& c, const Tensor& a, Scalar b){
    if(!c.track_grad()){
        ::nt::functional::fused_multiply_subtract_(c.detach(), a, b);
        return c;
    }
    handle_null_tensors(c, a, b, NT_INSIDE_FUNC_NAME);
    ::nt::functional::fused_multiply_subtract_(c.detach(), a, b);
    c.create_write_node([](const Tensor &grad, std::vector<intrusive_ptr<TensorGrad>> &parents){
        parents[0]->accumulate_gradient(grad);
    }, "FusedMultiplySubtract_");
    return c;
}

#undef NT_INSIDE_FUNC_NAME 

}
}
