#ifndef _NT_LAYER_REGISTRY_HPP_
#define _NT_LAYER_REGISTRY_HPP_
#include "../Module.h"
#include "custom_iterator.hpp"
#include "custom_iterator_map.hpp"
#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include <typeindex>

namespace nt{
namespace reflect{
namespace detail{

template<typename T, typename Arg>
inline constexpr decltype(auto) add_cv_like(Arg& arg) noexcept {
    if constexpr (std::is_const<T>::value && std::is_volatile<T>::value) {
        return const_cast<const volatile Arg&>(arg);
    }
    else if constexpr (std::is_const<T>::value) {
        return const_cast<const Arg&>(arg);
    }
    else if constexpr (std::is_volatile<T>::value) {
        return const_cast<volatile Arg&>(arg);
    }
    else {
        return const_cast<Arg&>(arg);
    }
}

template<typename T, typename Sb, typename Arg>
constexpr decltype(auto) workaround_cast(Arg& arg) noexcept {
    using output_arg_t = std::conditional_t<!std::is_reference<Sb>(), decltype(add_cv_like<T>(arg)), Sb>;
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
using RegistryEntry = std::tuple<std::type_index, 
			std::function<bool(Module*)>, 
			std::function<custom_any_iterator(Module*)>,
			std::function<custom_any_map(Module*)>,
			std::string,
			std::function<bool()>,
			std::function<bool()>,
			std::function<bool()>
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
inline ::nt::reflect::detail::custom_typed_map<T> get_named_module_vars(intrusive_ptr<Module>& ptr){
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return custom_typed_map<T>(std::get<3>(it)(ptr.get()));
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

inline bool forward_module_function_overriden(const intrusive_ptr<Module>& ptr_){
	intrusive_ptr<Module>& ptr = const_cast<intrusive_ptr<Module>&>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
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

inline bool eval_module_function_overriden(const intrusive_ptr<Module>& ptr_){
	intrusive_ptr<Module>& ptr = const_cast<intrusive_ptr<Module>&>(ptr_);
	for(auto& it : getRegistry()){
		if(std::get<1>(it)(ptr.get())){
			return std::get<7>(it)();
		}
	}
	return false;

}



}}}

#endif //_NT_LAYER_REGISTRY_HPP_
