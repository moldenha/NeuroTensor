#include "optional.h"

namespace nt{
namespace utils{

optional_tensor::optional_tensor(const optional_tensor& op)
	:tensor(op.tensor)
{}

optional_tensor::optional_tensor(optional_tensor&& op)
	:tensor(std::move(op.tensor))
{}

optional_tensor::optional_tensor(const intrusive_ptr<tensor_holder>& it)
	:tensor(it)
{}

optional_tensor::optional_tensor(intrusive_ptr<tensor_holder>&& it)
	:tensor(it)
{}


optional_tensor::optional_tensor(const Tensor& t)
	:tensor(make_intrusive<tensor_holder>(t))
{}

optional_tensor::optional_tensor(Tensor&& t)
	:tensor(make_intrusive<tensor_holder>(std::move(t)))
{}

optional_tensor::optional_tensor(std::nullptr_t)
	:tensor(nullptr)
{}

optional_tensor::optional_tensor()
	:tensor(nullptr)
{}


optional_tensor& optional_tensor::operator=(const optional_tensor& op){
	tensor = op.tensor;
	return *this;
}

optional_tensor& optional_tensor::operator=(optional_tensor&& op){
	tensor = std::move(op.tensor);
	return *this;
}

optional_tensor& optional_tensor::operator=(const Tensor& t){
	if(*this){
		tensor->tensor = t;
		return *this;
	}
	tensor = make_intrusive<tensor_holder>(t);
	return *this;
}

optional_tensor& optional_tensor::operator=(Tensor&& t){
	if(*this){
		tensor->tensor = std::move(t);
		return *this;
	}
	tensor = make_intrusive<tensor_holder>(std::move(t));
	return *this;
}

optional_tensor& optional_tensor::operator=(const intrusive_ptr<tensor_holder>& it){
	tensor = it;
	return *this;
}

optional_tensor& optional_tensor::operator=(intrusive_ptr<tensor_holder>&& it){
	tensor = std::move(it);
	return *this;
}


optional_tensor& optional_tensor::operator=(std::nullptr_t){
	if(*this){this->tensor.reset();}
	return *this;
}

optional_list::optional_list(const optional_list& op)
	:list(op.list)
{}

optional_list::optional_list(optional_list&& op)
	:list(std::move(op.list))
{}

optional_list::optional_list(const intrusive_ptr<intrusive_list<int64_t> >& it)
	:list(it)
{}

optional_list::optional_list(intrusive_ptr<intrusive_list<int64_t> >&& it)
	:list(it)
{}

optional_list::optional_list(std::initializer_list<int64_t> elements)
	:list(make_intrusive<intrusive_list<int64_t> >(elements))
{}


optional_list::optional_list()
	:list(nullptr)
{}


optional_list& optional_list::operator=(const optional_list& op){
	list = op.list;
	return *this;
}

optional_list& optional_list::operator=(optional_list&& op){
	list = std::move(op.list);
	return *this;
}

optional_list& optional_list::operator=(std::initializer_list<int64_t> elements){
	if(*this){
		(*list) = elements;
		return *this;
	}
	list = make_intrusive<intrusive_list<int64_t> >(elements);
	return *this;
}


optional_list& optional_list::operator=(const intrusive_ptr<intrusive_list<int64_t> >& it){
	list = it;
	return *this;
}

optional_list& optional_list::operator=(intrusive_ptr<intrusive_list<int64_t>>&& it){
	list = std::move(it);
	return *this;
}

//optional string
optional_string::optional_string(const optional_string& op)
	:str(op.str)
{}

optional_string::optional_string(optional_string&& op)
	:str(std::move(op.str))
{}

optional_string::optional_string(const intrusive_ptr<intrusive_string>& it)
	:str(it)
{}

optional_string::optional_string(intrusive_ptr<intrusive_string>&& it)
	:str(it)
{}


optional_string::optional_string(const std::string& t)
	:str(make_intrusive<intrusive_string>(t))
{}

optional_string::optional_string(std::string&& t)
	:str(make_intrusive<intrusive_string>(std::move(t)))
{}

optional_string::optional_string(std::nullptr_t)
	:str(nullptr)
{}

optional_string::optional_string()
	:str(nullptr)
{}


optional_string& optional_string::operator=(const optional_string& op){
	str = op.str;
	return *this;
}

optional_string& optional_string::operator=(optional_string&& op){
	str = std::move(op.str);
	return *this;
}

optional_string& optional_string::operator=(const std::string& t){
	if(*this){
		str->value = t;
		return *this;
	}
	str = make_intrusive<intrusive_string>(t);
	return *this;
}

optional_string& optional_string::operator=(std::string&& t){
	if(*this){
		str->value = std::move(t);
		return *this;
	}
	str = make_intrusive<intrusive_string>(std::move(t));
	return *this;
}

optional_string& optional_string::operator=(const intrusive_ptr<intrusive_string>& it){
	str = it;
	return *this;
}

optional_string& optional_string::operator=(intrusive_ptr<intrusive_string>&& it){
	str = std::move(it);
	return *this;
}


optional_string& optional_string::operator=(std::nullptr_t){
	if(*this){this->str.reset();}
	return *this;
}




}} //nt::utils::
