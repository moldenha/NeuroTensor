#include "TensorGrad.h"
#include "../Tensor.h"

namespace nt{
namespace layers{
namespace detail{


class LayerWrapper;

enum ValueType{
	Type_Scalar;
	Type_Tensor;
	Type_TensorGrad;
	Type_Layer;
};

struct TensorLayerValue{
	ValueType type;
	union Obj{
		Scalar     asScalar;
		Tensor     asTensor;
		TensorGrad asTensorGrad;
		LayerWrapper   asLayer;
	} o;

	TensorLayerValue& operator=(nt::Scalar vv){
		type = ValueType::Type_Scalar;
		o.asScalar = nt::Scalar(vv);
		return *this;
	}

	TensorLayerValue& operator=(nt::Tensor vv){
		type = ValueType::Type_Tensor;
		o.asTensor = vv;
		return *this;
	}
	
	TensorLayerValue& operator=(nt::TensorGrad vv){
		type = ValueType::Type_TensorGrad;
		o.asTensorGrad = vv;
		return *this;
	}

	TensorLayerValue& operator=(LayerWrapper vv){
		type = ValueType::Type_Layer;
		o.asLayer = vv;
		return *this;
	}



};


}


struct NSelf{
	std::unordered_map<std::string, TensorLayerValue> data;

    // Overload the subscript operator to allow setting values
    TypedValue& operator[](const std::string& key) {
        return data[key];
    }

    // Overload the subscript operator to allow getting values (const version)
    const TypedValue& operator[](const std::string& key) const {
        if (auto it = data.find(key); it != data.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Variable not found: " + key);
        }
    }

};


#define SET_VAR(cls, name, var)\
	cls->self[#name] = var

// Macro to access the value based on its type within a class
#define ACCESS_VAR(cls, name) \
    (cls->self[#name].type == Type_Scalar ? cls->self[#name].o.asScalar : \
    (cls->self[#name].type == Type_Tensor ? cls->self[#name].o.asTensor : \
    (cls->self[#name].type == Type_TensorGrad ? cls->self[#name].o.asTensorGrad : \
    (cls->self[#name].type == Type_Layer ? cls->self[#name].o.asLayer : \
    throw std::runtime_error("Unknown type")))))


class Layer : public intrusive_ptr_target{
	protected:
		NSelf self;
		constexpr std::string LayerName = "None";
		std::vector<std::pair<intrusive_ptr<TensorGrad>,
			intrusive_ptr<Layer> > > grads;
		//the order/ way it would go:
		//forward(TensorGrad){
		//
		//x *= a;
		//b = x + c;
		//d = layer1(b)
		//f = layer2(d)
		//z = f + q
		//o = z * 3
		//return o;
		//
		//}
		//the way that grads would be set up:
		//grads (in backward) would have:
		//({o, null}, {f, null}, {f_copy, layer2}, {b, layer1}, {d null}, {x, null})
		//at least something like that?
		//figure that out
	public:
		Layer() {}
		virtual TensorGrad forward(TensorGrad tg) {return tg;}
		virtual Tensor backward(Tensor last_grad, intrusive_ptr<TensorGrad> current_grad){
			current_grad->backward(last_grad);
			for(auto begin = grads.crbegin(); begin != grads.crend(); ++begin){
				
			}
		}


};

}}
