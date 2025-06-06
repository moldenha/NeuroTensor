#ifndef _NT_UTILS_OPTIONAL_STRING_H_
#define _NT_UTILS_OPTIONAL_STRING_H_

#include <string>
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include <initializer_list>
#include "optional_except.h"
#include "type_traits.h"


namespace nt{
namespace utils{

class intrusive_string: public intrusive_ptr_target{
	public:
		std::string value;
		intrusive_string() = default;
		intrusive_string(const intrusive_string& str):value(str.value) {}
		intrusive_string(intrusive_string&& str):value(std::move(str.value)) {}
		intrusive_string(const std::string& str):value(str) {}
		intrusive_string(std::string&& str):value(std::move(str)) {}
};


class optional_string{
	intrusive_ptr<intrusive_string> str;
	public:
		optional_string(const optional_string&);
		optional_string(optional_string&&);
		optional_string(const intrusive_ptr<intrusive_string>&);
		optional_string(intrusive_ptr<intrusive_string>&&);
		optional_string(const std::string&);
		optional_string(std::string&&);
		optional_string(std::nullptr_t);
		optional_string();
		optional_string& operator=(const optional_string&);
		optional_string& operator=(optional_string&&);
		optional_string& operator=(std::string&&);
		optional_string& operator=(const std::string&);
		optional_string& operator=(const intrusive_ptr<intrusive_string>&);
		optional_string& operator=(intrusive_ptr<intrusive_string>&&);
		optional_string& operator=(std::nullptr_t);
		
		inline bool has_value() const noexcept {return bool(str);}
		inline explicit operator bool() const noexcept {return has_value();}
		inline const std::string* operator->() const noexcept{
			if(!has_value()){return nullptr;}
			return &str->value;
		}
		inline std::string* operator->() noexcept{
			if(!has_value()){return nullptr;}
			return &str->value;
		}
		inline const std::string& operator*() const& noexcept{
			return str->value;
		}
		inline std::string& operator*() & noexcept{
			return str->value;
		}

		inline const std::string& value() const&{
			if(!has_value()){throw bad_optional_access();}
			return str->value;
		}
		inline std::string& value() &{
			if(!has_value()){throw bad_optional_access();}
			return str->value;
		}

		template<class U>
		inline std::string value_or(U&& default_value) const& {
			return bool(*this) ? **this : std::string(std::forward<U>(default_value));
		}
		template<class U>
		inline std::string value_or(U&& default_value) && {
			return bool(*this) ? std::move(**this) : std::string(std::forward<U>(default_value));
		}

		template<class F>
		inline auto and_then(F&& f) &{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, std::string&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) const&{
			if (*this)
			    return std::invoke(std::forward<F>(f), **this);
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, const std::string&>>{};
		}

		template<class F>
		inline auto and_then(F&& f) &&{
			if (*this)
			    return std::invoke(std::forward<F>(f), std::move(**this));
			else
			    return std::remove_cvref_t<std::invoke_result_t<F, std::string>>{};
		}

		inline void swap(optional_string& op) noexcept {op.str.swap(str);}
		inline void reset(){str.reset(); str = nullptr;}
		template<typename... Args>
		inline std::string& emplace(Args&&... args){
			if(*this){
				reset();
				return emplace(std::forward<Args&&>(args)...);
			}
			this->str = make_intrusive<intrusive_string>(std::string(std::forward<Args&&>(args)...));
			return this->str->value;
		}

};



}} //nt::utils::

namespace std{
inline void swap(nt::utils::optional_string& a, nt::utils::optional_string& b) noexcept {a.swap(b);}
} //std::

#endif
