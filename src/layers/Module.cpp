#include "Module.h"

namespace nt{
namespace layers{

AttributeAccessTensor::AttributeAccessTensor(const AttributeAccessTensor& att){
	for(const auto& pair : att.attributes){
		attributes[pair.first] = pair.second;
	}
}

AttributeAccessTensor::AttributeAccessTensor(AttributeAccessTensor&& att)
	:attributes(std::move(att.attributes))
{}

AttributeAccessTensor& AttributeAccessTensor::operator=(AttributeAccessTensor&& att){
	attributes = std::move(att.attributes);
}

AttributeAccessTensor& AttributeAccessTensor::operator=(const AttributeAccessTensor& att){
	AttributeAccessTensor cpy(att);
	*this = std::move(cpy);
	return *this;
}

std::vector<std::reference_wrapper<Tensor>> AttributeAccessTensor::get_parameters(){
	std::vector<std::reference_wrapper<Tensor> > params;
	params.reserve(attributes.size());
	for(auto& pair : attributes){params.push_back(std::ref(*pair.second));}
	return std::move(params);
}

std::vector<std::reference_wrapper<const Tensor>> AttributeAccessTensor::get_parameters() const {
	std::vector<std::reference_wrapper<const Tensor> > params;
	params.reserve(attributes.size());
	for(auto& pair : attributes){params.push_back(std::cref(*pair.second));}
	return std::move(params);
}

std::map<std::string, std::reference_wrapper<Tensor> > AttributeAccessTensor::get_parameter_map(){
	std::map<std::string, std::reference_wrapper<Tensor> > Map;
	for(auto& pair : attributes){Map.insert({pair.first,std::ref(*pair.second)});}
	return std::move(Map);
}

std::map<std::string, std::reference_wrapper<const Tensor> > AttributeAccessTensor::get_parameter_map() const{
	std::map<std::string, std::reference_wrapper<const Tensor> > Map;
	for(auto& pair : attributes){Map.insert({pair.first,std::cref(*pair.second)});}
	return std::move(Map);
}


void AttributeAccessTensor::to_dtype(DType dt){
	for(auto& pair : attributes){
		Tensor& t = *pair.second;
		Tensor n = t.to_dtype(dt);
		t.swap(n);
	}
}



AttributeAccessLayer::AttributeAccessLayer(const AttributeAccessLayer& att){
	for(const auto& pair : att.attributes){
		attributes[pair.first] = pair.second;
	}

}
AttributeAccessLayer::AttributeAccessLayer(AttributeAccessLayer&& att)
	:attributes(std::move(att.attributes))
{}
AttributeAccessLayer::AttributeAccessLayer& operator=(const AttributeAccessLayer&){
	AttributeAccessLayer cpy(att);
	*this = std::move(cpy);
	return *this;

}
AttributeAccessLayer::AttributeAccessLayer& operator=(AttributeAccessLayer&& ){
	attributes = std::move(att.attributes);
}

std::vector<std::reference_wrapper<AttributeAccessTensor>> AttributeAccessLayer::get_layers(){
	std::vector<std::reference_wrapper<AttributeAccessTensor>> vec;
	vec.reserve(size());
	for(auto& pair : attributes){
		vec.push_back(std::ref(pair.second));
	}
	return std::move(vec);
}

std::vector<std::reference_wrapper<const AttributeAccessTensor>> AttributeAccessLayer::get_layers() const{
	std::vector<std::reference_wrapper<const AttributeAccessTensor>> vec;
	vec.reserve(size());
	for(const auto& pair : attributes){
		vec.push_back(std::cref(pair.second));
	}
	return std::move(vec);
}

std::map<std::string, std::reference_wrapper<AttributeAccessTensor>> AttributeAccessLayer::get_layer_map() {
	std::map<std::string, std::reference_wrapper<AttributeAccessTensor>> Map;
	for(auto& pair : attributes){
		Map[pair.first] = std::ref(pair.second);
	}
	return std::move(Map);
}

std::map<std::string, std::reference_wrapper<const AttributeAccessTensor>> AttributeAccessLayer::get_layer_map() const {
	std::map<std::string, std::reference_wrapper<const AttributeAccessTensor>> Map;
	for(const auto& pair : attributes){
		Map[pair.first] = std::cref(pair.second);
	}
	return std::move(Map);
}

void AttributeAccessLayer::to_dtype(DType dt){
	for(auto& pair : attributes){
		pair.second->to_dtype(dt);
	}
}

Tensor AttributeAccessLayer::parameters(){
	std::vector<Tensor> ts;
	ts.reserve(size());
	for(auto& pair : attributes)
		ts.push_back(ts.second->parameters());
	return functional::stack(ts);
}

}
}
