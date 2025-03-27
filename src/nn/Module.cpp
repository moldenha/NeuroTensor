#include "Module.h"
#include "Layer.h"

namespace nt {

void Module::register_parameter(std::string name, TensorGrad &tg) {
    auto result = this->_mapped_grad_names_wrapped.insert({name, std::ref(tg)});
    utils::throw_exception(result.second,
                           "Key parameter of name $ already exists", name);
}
void Module::register_module(std::string name, Layer &l) {
    auto result = this->_mapped_layer_names_wrapped.insert({name, std::ref(l)});
    utils::throw_exception(result.second, "Key layer of name $ already exists",
                           name);
}

std::map<std::string, std::reference_wrapper<TensorGrad>> &
Module::_get_mapped_grad_names_wrapped() noexcept {
    return _mapped_grad_names_wrapped;
}

std::map<std::string, std::reference_wrapper<Layer>> &
Module::_get_mapped_layer_names_wrapped() noexcept {
    return _mapped_layer_names_wrapped;
}


std::string Module::name() const noexcept {
    return reflect::detail::get_module_name(this);
}

reflect::detail::custom_typed_iterator<Layer> Module::get_all_layers() {
    reflect::detail::custom_typed_map<Layer> named =
        this->get_all_named_layers();
    return reflect::detail::custom_typed_iterator<Layer>(
        named.get_references().begin(), named.get_references().end());
}

reflect::detail::custom_typed_map<Layer> Module::get_all_named_layers() {
    reflect::detail::custom_typed_map<Layer> my_layers =
        reflect::detail::get_named_module_vars<Layer>(this);
    my_layers.extend(this->_mapped_layer_names_wrapped);
    std::string add = this->name();
    for(auto it : my_layers){
        const std::string& name = it.first;
        my_layers.extend_unique(it.second.get_all_named_layers(), add+ '.' +name);
    }
    return std::move(my_layers);
}

reflect::detail::custom_typed_iterator<TensorGrad> Module::parameters() {
    reflect::detail::custom_typed_map<TensorGrad> named =
        this->named_parameters();
    return reflect::detail::custom_typed_iterator<TensorGrad>(
        named.get_references().begin(), named.get_references().end());
}

reflect::detail::custom_typed_map<TensorGrad> Module::named_parameters() {
    reflect::detail::custom_typed_map<TensorGrad> start =
        reflect::detail::get_named_module_vars<TensorGrad>(this);
    start.extend(this->_mapped_grad_names_wrapped);

    for (const auto &l : reflect::detail::get_named_module_vars<Layer>(this)) {
        start.extend_unique(std::move(l.second.named_parameters()),
                            '.' + l.first);
    }
    return std::move(start);
}

// AttributeAccessTensor::AttributeAccessTensor(const AttributeAccessTensor&
// att){ 	for(const auto& pair : att.attributes){ 		attributes[pair.first] =
// pair.second;
// 	}
// }

// AttributeAccessTensor::AttributeAccessTensor(AttributeAccessTensor&& att)
// 	:attributes(std::move(att.attributes))
// {}

// AttributeAccessTensor&
// AttributeAccessTensor::operator=(AttributeAccessTensor&& att){ 	attributes =
// std::move(att.attributes);
// }

// AttributeAccessTensor& AttributeAccessTensor::operator=(const
// AttributeAccessTensor& att){ 	AttributeAccessTensor cpy(att); 	*this =
// std::move(cpy); 	return *this;
// }

// std::vector<std::reference_wrapper<Tensor>>
// AttributeAccessTensor::get_parameters(){
// 	std::vector<std::reference_wrapper<Tensor> > params;
// 	params.reserve(attributes.size());
// 	for(auto& pair : attributes){params.push_back(std::ref(*pair.second));}
// 	return std::move(params);
// }

// std::vector<std::reference_wrapper<const Tensor>>
// AttributeAccessTensor::get_parameters() const {
// 	std::vector<std::reference_wrapper<const Tensor> > params;
// 	params.reserve(attributes.size());
// 	for(auto& pair : attributes){params.push_back(std::cref(*pair.second));}
// 	return std::move(params);
// }

// std::map<std::string, std::reference_wrapper<Tensor> >
// AttributeAccessTensor::get_parameter_map(){ 	std::map<std::string,
// std::reference_wrapper<Tensor> > Map; 	for(auto& pair :
// attributes){Map.insert({pair.first,std::ref(*pair.second)});} 	return
// std::move(Map);
// }

// std::map<std::string, std::reference_wrapper<const Tensor> >
// AttributeAccessTensor::get_parameter_map() const{ 	std::map<std::string,
// std::reference_wrapper<const Tensor> > Map; 	for(auto& pair :
// attributes){Map.insert({pair.first,std::cref(*pair.second)});} 	return
// std::move(Map);
// }

// void AttributeAccessTensor::to_dtype(DType dt){
// 	for(auto& pair : attributes){
// 		Tensor& t = *pair.second;
// 		Tensor n = t.to_dtype(dt);
// 		t.swap(n);
// 	}
// }

// AttributeAccessLayer::AttributeAccessLayer(const AttributeAccessLayer& att){
// 	for(const auto& pair : att.attributes){
// 		attributes[pair.first] = pair.second;
// 	}

// }
// AttributeAccessLayer::AttributeAccessLayer(AttributeAccessLayer&& att)
// 	:attributes(std::move(att.attributes))
// {}
// AttributeAccessLayer::AttributeAccessLayer& operator=(const
// AttributeAccessLayer&){ 	AttributeAccessLayer cpy(att); 	*this =
// std::move(cpy); 	return *this;

// }
// AttributeAccessLayer::AttributeAccessLayer& operator=(AttributeAccessLayer&&
// ){ 	attributes = std::move(att.attributes);
// }

// std::vector<std::reference_wrapper<AttributeAccessTensor>>
// AttributeAccessLayer::get_layers(){
// 	std::vector<std::reference_wrapper<AttributeAccessTensor>> vec;
// 	vec.reserve(size());
// 	for(auto& pair : attributes){
// 		vec.push_back(std::ref(pair.second));
// 	}
// 	return std::move(vec);
// }

// std::vector<std::reference_wrapper<const AttributeAccessTensor>>
// AttributeAccessLayer::get_layers() const{
// 	std::vector<std::reference_wrapper<const AttributeAccessTensor>> vec;
// 	vec.reserve(size());
// 	for(const auto& pair : attributes){
// 		vec.push_back(std::cref(pair.second));
// 	}
// 	return std::move(vec);
// }

// std::map<std::string, std::reference_wrapper<AttributeAccessTensor>>
// AttributeAccessLayer::get_layer_map() { 	std::map<std::string,
// std::reference_wrapper<AttributeAccessTensor>> Map; 	for(auto& pair :
// attributes){ 		Map[pair.first] = std::ref(pair.second);
// 	}
// 	return std::move(Map);
// }

// std::map<std::string, std::reference_wrapper<const AttributeAccessTensor>>
// AttributeAccessLayer::get_layer_map() const { 	std::map<std::string,
// std::reference_wrapper<const AttributeAccessTensor>> Map; 	for(const auto&
// pair : attributes){ 		Map[pair.first] = std::cref(pair.second);
// 	}
// 	return std::move(Map);
// }

// void AttributeAccessLayer::to_dtype(DType dt){
// 	for(auto& pair : attributes){
// 		pair.second->to_dtype(dt);
// 	}
// }

// Tensor AttributeAccessLayer::parameters(){
// 	std::vector<Tensor> ts;
// 	ts.reserve(size());
// 	for(auto& pair : attributes)
// 		ts.push_back(ts.second->parameters());
// 	return functional::stack(ts);
// }

} // namespace nt
