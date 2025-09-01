#ifndef NT_UTILS_OPTIONAL_TENSORGRAD_H__
#define NT_UTILS_OPTIONAL_TENSORGRAD_H__

#include "../nn/TensorGrad.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include <initializer_list>
#include "optional_except.h"
#include "type_traits.h"
#include "api_macro.h"

namespace nt{
namespace utils{


class NEUROTENSOR_API optional_tensorgrad{
	intrusive_ptr<TensorGrad> tensor;
	public:
		optional_tensorgrad(const optional_tensorgrad& other):tensor(other.tensor) {}
		optional_tensorgrad(optional_tensorgrad&& other):tensor(std::move(other.tensor)) {}
		optional_tensorgrad(const intrusive_ptr<TensorGrad>& other):tensor(other) {}
		optional_tensorgrad(intrusive_ptr<TensorGrad>&& other):tensor(std::move(other)) {}
		optional_tensorgrad(const TensorGrad& other):tensor(make_intrusive<TensorGrad>(other)) {}
		optional_tensorgrad(TensorGrad&& other):tensor(make_intrusive<TensorGrad>(std::move(other))) {}
		optional_tensorgrad(std::nullptr_t):tensor(nullptr) {}
		optional_tensorgrad():tensor(nullptr) {}
		inline optional_tensorgrad& operator=(const optional_tensorgrad& other){tensor = other.tensor; return *this;}
		inline optional_tensorgrad& operator=(optional_tensorgrad&& other){tensor = std::move(other.tensor); return *this;}
		inline optional_tensorgrad& operator=(TensorGrad&& other){tensor = make_intrusive<TensorGrad>(std::move(other)); return *this;}
		inline optional_tensorgrad& operator=(const TensorGrad& other){tensor = make_intrusive<TensorGrad>(other); return *this;}
		inline optional_tensorgrad& operator=(const intrusive_ptr<TensorGrad>& other){tensor = other; return *this;}
		inline optional_tensorgrad& operator=(intrusive_ptr<TensorGrad>&& other){tensor = std::move(other); return *this;}
		inline optional_tensorgrad& operator=(std::nullptr_t){tensor.reset(); tensor = nullptr; return *this;}
		
		inline bool has_value() const noexcept {return bool(tensor);}
		inline explicit operator bool() const noexcept {return has_value();}
		inline const TensorGrad* operator->() const noexcept{
			if(!has_value()){return nullptr;}
			return &(*tensor);
		}
		inline TensorGrad* operator->() noexcept{
			if(!has_value()){return nullptr;}
			return &(*tensor);
		}
		inline const TensorGrad& operator*() const& noexcept{
			return *tensor;
		}
		inline TensorGrad& operator*() & noexcept{
			return *tensor;
		}

		inline const TensorGrad& value() const&{
			if(!has_value()){throw bad_optional_access();}
			return *tensor;
		}
		inline TensorGrad& value() &{
			if(!has_value()){throw bad_optional_access();}
			return *tensor;
		}

		template<class U>
		inline TensorGrad value_or(U&& default_value) const& {
			return bool(*this) ? **this : TensorGrad(std::forward<U>(default_value));
		}
		template<class U>
		inline TensorGrad value_or(U&& default_value) && {
			return bool(*this) ? std::move(**this) : TensorGrad(std::forward<U>(default_value));
		}

		template<class F>
		inline auto and_then(F&& f) &{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return ::nt::type_traits::remove_cvref_t<std::invoke_result_t<F, TensorGrad&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) const&{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return ::nt::type_traits::remove_cvref_t<std::invoke_result_t<F, const TensorGrad&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) &&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return ::nt::type_traits::remove_cvref_t<std::invoke_result_t<F, TensorGrad>>{};
		}

		inline void swap(optional_tensorgrad& op) noexcept {op.tensor.swap(tensor);}
		inline void reset(){tensor.reset(); tensor = nullptr;}
		template<typename... Args>
		inline TensorGrad& emplace(Args&&... args){
			if(*this){
				reset();
				return emplace(std::forward<Args&&>(args)...);
			}
			this->tensor = make_intrusive<TensorGrad>(TensorGrad(std::forward<Args&&>(args)...));
			return *this->tensor;
		}

};



}} //nt::utils::

namespace std{
inline void swap(nt::utils::optional_tensorgrad& a, nt::utils::optional_tensorgrad& b) noexcept {a.swap(b);}
} //std::

#endif
