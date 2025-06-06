#ifndef _NT_UTILS_OPTIONAL_TENSOR_H_
#define _NT_UTILS_OPTIONAL_TENSOR_H_

#include "tensor_holder.h"
#include "../Tensor.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include <initializer_list>
#include "optional_except.h"
#include "type_traits.h"

namespace nt{
namespace utils{


class optional_tensor{
	intrusive_ptr<tensor_holder> tensor;
	public:
		optional_tensor(const optional_tensor&);
		optional_tensor(optional_tensor&&);
		optional_tensor(const intrusive_ptr<tensor_holder>&);
		optional_tensor(intrusive_ptr<tensor_holder>&&);
		optional_tensor(const Tensor&);
		optional_tensor(Tensor&&);
		optional_tensor(std::nullptr_t);
		optional_tensor();
		optional_tensor& operator=(const optional_tensor&);
		optional_tensor& operator=(optional_tensor&&);
		optional_tensor& operator=(Tensor&&);
		optional_tensor& operator=(const Tensor&);
		optional_tensor& operator=(const intrusive_ptr<tensor_holder>&);
		optional_tensor& operator=(intrusive_ptr<tensor_holder>&&);
		optional_tensor& operator=(std::nullptr_t);
		
		inline bool has_value() const noexcept {return bool(tensor);}
		inline explicit operator bool() const noexcept {return has_value();}
		inline const Tensor* operator->() const noexcept{
			if(!has_value()){return nullptr;}
			return &tensor->tensor;
		}
		inline Tensor* operator->() noexcept{
			if(!has_value()){return nullptr;}
			return &tensor->tensor;
		}
		inline const Tensor& operator*() const& noexcept{
			return tensor->tensor;
		}
		inline Tensor& operator*() & noexcept{
			return tensor->tensor;
		}

		inline const Tensor& value() const&{
			if(!has_value()){throw bad_optional_access();}
			return tensor->tensor;
		}
		inline Tensor& value() &{
			if(!has_value()){throw bad_optional_access();}
			return tensor->tensor;
		}

		template<class U>
		inline Tensor value_or(U&& default_value) const& {
			return bool(*this) ? **this : Tensor(std::forward<U>(default_value));
		}
		template<class U>
		inline Tensor value_or(U&& default_value) && {
			return bool(*this) ? std::move(**this) : Tensor(std::forward<U>(default_value));
		}

		template<class F>
		inline auto and_then(F&& f) &{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, Tensor&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) const&{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, const Tensor&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) &&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, Tensor>>{};
		}

		inline void swap(optional_tensor& op) noexcept {op.tensor.swap(tensor);}
		inline void reset(){tensor.reset(); tensor = nullptr;}
		template<typename... Args>
		inline Tensor& emplace(Args&&... args){
			if(*this){
				reset();
				return emplace(std::forward<Args&&>(args)...);
			}
			this->tensor = make_intrusive<tensor_holder>(Tensor(std::forward<Args&&>(args)...));
			return this->tensor->tensor;
		}

};



}} //nt::utils::

namespace std{
inline void swap(nt::utils::optional_tensor& a, nt::utils::optional_tensor& b) noexcept {a.swap(b);}
} //std::

#endif
