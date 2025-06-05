#ifndef __NT_NN_SCALAR_GRAD_H__
#define __NT_NN_SCALAR_GRAD_H__

namespace nt{
class ScalarGrad;
}

#include "../Tensor.h"
#include "../dtype/Scalar.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "TensorGrad.h"

namespace nt{


//this is meant to hold a gradient and a scalar
//any operation that happens to the scalar, also happens to the gradient
class ScalarGrad{
    Scalar item;
    intrusive_ptr<TensorGrad> parent;
    public:
        Tensor grad; 
        ScalarGrad(Scalar _item, Tensor _grad, intrusive_ptr<TensorGrad> _parent);
        ScalarGrad(Scalar _item, Tensor _grad, TensorGrad _parent);
        ScalarGrad() = delete;
        inline bool isComplex() const {return item.isComplex();}
		inline bool isFloatingPoint() const {return item.isFloatingPoint();}
		inline bool isIntegral() const {return item.isIntegral();}
		inline bool isBoolean() const {return item.isBoolean();}
		inline bool isZero() const {return item.isZero();}
		inline bool isNegative() const {return item.isNegative();}
        inline bool isInfinity() const {return item.isInfinity();}
        inline bool isNan() const {return item.isNan();}

        inline ScalarGrad operator+(const Scalar& s) const {return ScalarGrad(item + s, grad + s, parent);}
		inline ScalarGrad operator-(const Scalar& s) const {return ScalarGrad(item - s, grad - s, parent);} 
		inline ScalarGrad operator/(const Scalar& s) const {return ScalarGrad(item / s, grad / s, parent);} 
		inline ScalarGrad operator*(const Scalar& s) const {return ScalarGrad(item * s, grad * s, parent);}

		inline ScalarGrad& operator+=(const Scalar& s) {item += s; grad += s; return *this;}
		inline ScalarGrad& operator-=(const Scalar& s) {item -= s; grad -= s; return *this;} 
		inline ScalarGrad& operator/=(const Scalar& s) {item /= s; grad /= s; return *this;} 
		inline ScalarGrad& operator*=(const Scalar& s) {item *= s; grad *= s; return *this;} 

        inline ScalarGrad& operator=(const Scalar& s) {item = s; grad = s; return *this;}
        inline ScalarGrad operator-() const {return ScalarGrad(-item, -grad, parent);}
        inline ScalarGrad inverse() const {return ScalarGrad(item.inverse(), grad.inverse(), parent);}
        
        inline DType dtype() const {return grad.dtype;}

        ScalarGrad to(DType);
        void backward();
        friend std::ostream& operator<<(std::ostream&, const ScalarGrad&);
        

};

}

#endif


