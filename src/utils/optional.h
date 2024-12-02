#ifndef _UTILS_OPTIONAL_H_
#define _UITLS_OPTIONAL_H_

#include "../Tensor.h"
#include "../layers/TensorGrad.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{
namespace utils{


class bad_optional_access : public std::exception {
public:
    const char* what() const noexcept override {
        return "bad_optional_access";
    }
};


class optional_tensor{
	intrusive_ptr<tensor_holder> tensor;
	public:
		optional_tensor(const optional_tensor&);
		optional_tensor(optional_tensor&&);
		optional_tensor(const intrusive_ptr<tensor_holder>&);
		optional_tensor(intrusive_ptr<tensor_holder>&);
		optional_tensor(const Tensor&);
		optional_tensor(Tensor&&);
		optional_tensor(nullptr);
		optional_tensor();
		optional_tensor& operator=(const optional_tensor&);
		optional_tensor& operator=(optional_tensor&&);
		optional_tensor& operator=(const Tensor&);
		optional_tensor& operator=(Tensor&&);
		optional_tensor& operator=(const intrusive_ptr<tensor_holder>);
		optional_tensor& operator=(intrusive_ptr<tensor_holder>&&);
		optional_tensor& operator=(nullptr);
		
		inline constexpr bool has_value() const noexcept {return tensor != nullptr;}
		inline constexpr explicit operator bool() const noexcept {return has_value();}
		inline constexpr const Tensor* operator->() const noexcept{
			if(!has_value()){return nullptr;}
			return &tensor->tensor;
		}
		inline constexpr Tensor* operator->() noexcept{
			if(!has_value()){return nullptr;}
			return &tensor->tensor;
		}
		inline constexpr const Tensor& operator*() const& noexcept{
			return tensor->tensor;
		}
		inline constexpr Tensor& operator*() & noexcept{
			return tensor->tensor;
		}
		inline constexpr const Tensor&& operator*() const&& noexcept {
			return tensor->tensor;
		}
		inline constexpr Tensor&& operator*() && noexcept {
			return tensor->tensor;
		}

		inline constexpr const Tensor& value() const&{
			if(!has_value(){throw bad_optional_access();}
			return tensor->tensor;
		}
		inline constexpr Tensor& value() &{
			if(!has_value(){throw bad_optional_access();}
			return tensor->tensor;
		}
		inline constexpr const Tensor&& value() const&&{
			if(!has_value(){throw bad_optional_access();}
			return tensor->tensor;
		}
		inline constexpr Tensor&& value() &&{
			if(!has_value(){throw bad_optional_access();}
			return tensor->tensor;
		}

		template<class U>
		inline constexpr Tensor value_or(U&& default_value) const& {
			return bool(*this) ? **this : Tensor(std::forward<U>(default_value))
		}
		template<class U>
		inline constexpr Tensor value_or(U&& default_value) && {
			return bool(*this) ? std::move(**this) : Tensor(std::forward<U>(default_value))
		}

		template<class F>
		inline constexpr auto and_then(F&& f) &{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, Tensor&>>{};
		}

		template<class F>
		inline constexpr auto and_then(F&& f) const&{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, const Tensor&>>{};
		}

		template<class F>
		inline constexpr auto and_then(F&& f) &&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, Tensor>>{};
		}

		template<class F>
		inline constexpr auto and_then(F&& f) const&&{
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
inline void swap(optional_tensor& a, optional_tensor& b) noexcept {a.swap(b);}
} //std::

#endif