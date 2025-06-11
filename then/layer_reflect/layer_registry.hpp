#ifndef _NT_LAYER_REGISTRY_HPP_
#define _NT_LAYER_REGISTRY_HPP_
#include "../../nn/Module.h"
#include "custom_iterator.hpp"
#include "custom_iterator_map.hpp"
#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include <typeindex>
#include <any>
#include "../../utils/type_traits.h"
#include "../../utils/utils.h"
#include "../../utils/any_ref.h"

namespace nt{
namespace reflect{
namespace detail{


template<typename T, typename Sb, typename Arg>
inline constexpr decltype(auto) workaround_cast(Arg& arg) noexcept {
    using output_arg_t = std::conditional_t<!std::is_reference<Sb>(), decltype(utils::add_cv_like<T>(arg)), Sb>;
    return const_cast<output_arg_t>(arg);
}

//this might be a good place for std::any
//type_index
//get if this is the type
//get all variables into a single iterator
//get all variables attached to their names into a single iterator
//the name of the class
//get if the forward function has been overriden
//get if the backward function has been overriden
//get if the eval function has been overriden
//get the backward function to register on a tensor grad
using RegistryEntry = std::tuple<std::type_index, 
			std::function<bool(Module*)>, 
			std::function<custom_any_iterator(Module*)>,
			std::function<custom_any_map(Module*)>,
			std::string,
			std::function<bool()>,
			std::function<bool()>,
			std::function<bool()>,
			std::function<std::function<void(const Tensor&, intrusive_ptr<TensorGrad>)>(Module*)>,
			std::function<TensorGrad(Module*, std::vector<utils::any_ref>)>,
			std::function<TensorGrad(Module*, std::vector<utils::any_ref>)>
>;

using RegistryType = std::vector<RegistryEntry>;



inline RegistryType& getRegistry() {
    static RegistryType registry;
    return registry;
}


//functions for layer registry

inline bool is_instance_of(intrusive_ptr<Module>& ptr, std::type_index type) noexcept {
	for(auto& it : getRegistry()){
		std::type_index& t = std::get<0>(it);
		if(t == type){
			return std::get<1>(it)(ptr.get());
		}
	}
	return false;
}

inline bool is_instance_of(Module* ptr, std::type_index type) noexcept {
	for(auto& it : getRegistry()){
		std::type_index& t = std::get<0>(it);
		if(t == type){
			return std::get<1>(it)(ptr);
		}
	}
	return false;
}



template<typename T>
inline ::nt::reflect::detail::custom_typed_iterator<T> get_module_vars(intrusive_ptr<Module>& ptr){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return custom_typed_iterator<T>(std::get<2>(it)(ptr.get()));
		}
	}
	return custom_typed_iterator<T>();

}

template<typename T>
inline ::nt::reflect::detail::custom_typed_iterator<T> get_module_vars(Module* ptr){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return custom_typed_iterator<T>(std::get<2>(it)(ptr));
		}
	}
	return custom_typed_iterator<T>();

}

template<typename T>
inline ::nt::reflect::detail::custom_typed_map<T> get_named_module_vars(intrusive_ptr<Module>& ptr){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return custom_typed_map<T>(std::get<3>(it)(ptr.get()));
		}
	}
	return custom_typed_map<T>();
}

template<typename T>
inline ::nt::reflect::detail::custom_typed_map<T> get_named_module_vars(Module* ptr){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return custom_typed_map<T>(std::get<3>(it)(ptr));
		}
	}
	return custom_typed_map<T>();
}

inline std::string get_module_name(const intrusive_ptr<Module>& ptr_) noexcept {
	intrusive_ptr<Module>& ptr = const_cast<intrusive_ptr<Module>&>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<4>(it);
		}
	}
	return "Module";

}

inline std::string get_module_name(const Module* ptr_) noexcept {
	Module* ptr = const_cast<Module*>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return std::get<4>(it);
		}
	}
	return "Module";

}

inline bool forward_module_function_overriden(const intrusive_ptr<Module>& ptr_){
	intrusive_ptr<Module>& ptr = const_cast<intrusive_ptr<Module>&>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<5>(it)();
		}
	}
	return false;

}

inline bool forward_module_function_overriden(const Module* ptr_){
	Module* ptr = const_cast<Module*>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return std::get<5>(it)();
		}
	}
	return false;

}

inline bool backward_module_function_overriden(const intrusive_ptr<Module>& ptr_){
	intrusive_ptr<Module>& ptr = const_cast<intrusive_ptr<Module>&>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<6>(it)();
		}
	}
	return false;

}

inline bool backward_module_function_overriden(const Module* ptr_){
	Module* ptr = const_cast<Module*>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return std::get<6>(it)();
		}
	}
	return false;

}

inline bool eval_module_function_overriden(const intrusive_ptr<Module>& ptr_){
	intrusive_ptr<Module>& ptr = const_cast<intrusive_ptr<Module>&>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<7>(it)();
		}
	}
	return false;
}

inline bool eval_module_function_overriden(const Module* ptr_){
	Module* ptr = const_cast<Module*>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return std::get<7>(it)();
		}
	}
	return false;
}

inline std::function<void(const Tensor&, intrusive_ptr<TensorGrad>)> get_backward_function(intrusive_ptr<Module>& ptr){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<8>(it)(ptr.get());
		}
	}
	return nullptr;	
}

inline std::function<void(const Tensor&, intrusive_ptr<TensorGrad>)> get_backward_function(Module* ptr){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return std::get<8>(it)(ptr);
		}
	}
	return nullptr;	
}

inline TensorGrad run_forward_function(intrusive_ptr<Module>& ptr, std::vector<utils::any_ref> vec_){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<9>(it)(ptr.get(), std::move(vec_));
		}
	}
    utils::throw_exception(false, "eror, layer not found when running forward, check to make sure all layers were registered!");
	return TensorGrad(nullptr);	
}

inline TensorGrad run_forward_function(Module* ptr, std::vector<utils::any_ref> vec_){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr)){
			return std::get<9>(it)(ptr, std::move(vec_));
		}
	}
    utils::throw_exception(false, "eror, layer not found when running forward, check to make sure all layers were registered!");
	return TensorGrad(nullptr);	
}

inline TensorGrad run_eval_function(intrusive_ptr<Module>& ptr, std::vector<utils::any_ref> vec_){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<10>(it)(ptr.get(), std::move(vec_));
		}
	}
	return TensorGrad(nullptr);	
}



}}}

#endif //_NT_LAYER_REGISTRY_HPP_
