#ifndef _NT_FUNCTIONAL_H_
#define _NT_FUNCTIONAL_H_
#include "../Tensor.h"


#include <vector>
#include <functional>
#include <optional>
#include "functional_matmult.h"
#include "functional_fold.h"
#include "functional_conv.h"
#include "../utils/optional_list.h"
#include "../dtype/DType.h"
#include <cstring>
#include "numpy.h"

namespace nt{
namespace functional{

Tensor zeros(SizeRef, DType dt=DType::Float);
Tensor zeros_like(const Tensor&);
Tensor ones(SizeRef, DType dt=DType::Float);
Tensor ones_like(const Tensor&);
Tensor nums(SizeRef, const Scalar, DType dt = DType::Float);
Tensor randint(Scalar, Scalar, SizeRef, DType dt = DType::int32);
Tensor rand(Scalar, Scalar, SizeRef, DType dt = DType::Float32);
Tensor randn(SizeRef, DType dt = DType::Float);
Tensor randbools(SizeRef, double);//the percentage between 0 and 1 that are true
void xavier_uniform_(Tensor&);
Tensor cat(const Tensor& _a, const Tensor &_b, int8_t dim=0);
Tensor hadamard_multiply(const Tensor&, const Tensor&);
Tensor& hadamard_multiply_this(Tensor&, const Tensor&);
Tensor add(const Tensor&, const Tensor&);
Tensor& add_(Tensor&, const Tensor&);
Tensor subtract(const Tensor&, const Tensor&);
Tensor& subtract_(Tensor&, const Tensor&);
Tensor divide(const Tensor&, const Tensor&);
Tensor& divide_(Tensor&, const Tensor&);
//returns c + (a * b);
Tensor fused_multiply_add(const Tensor& c, const Tensor& a, const Tensor& b);
Tensor fused_multiply_add(const Tensor& c, const Tensor& a, Scalar b);
//returns c += (a * b);
Tensor& fused_multiply_add_(Tensor& c, const Tensor& a, const Tensor& b);
Tensor& fused_multiply_add_(Tensor& c, const Tensor& a, Scalar b);
//returns c - (a * b);
Tensor fused_multiply_subtract(const Tensor& c, const Tensor& a, const Tensor& b);
Tensor fused_multiply_subtract(const Tensor& c, const Tensor& a, Scalar b);
//returns c -= (a * b);
Tensor& fused_multiply_subtract_(Tensor& c, const Tensor& a, const Tensor& b);
Tensor& fused_multiply_subtract_(Tensor& c, const Tensor& a, Scalar b);


Tensor dot(const Tensor&, const Tensor&, utils::optional_list dim = nullptr, bool keepdim = false);

Tensor arange(typename Tensor::size_value_t total_size, DType dt = DType::Float, Scalar start = 0);
Tensor arange(SizeRef, DType dt = DType::Float, Scalar start = 0);
bool all(const Tensor&);
bool any(const Tensor&);
Tensor all(const Tensor, int64_t dim);
Tensor any(const Tensor, int64_t dim);
void save(const Tensor&, const char*);
Tensor load(const char*);

Tensor conv2dT(const Tensor& image, const Tensor& kernel, const uint32_t s_h=1, const uint32_t s_w=1);
void softmax_(Tensor&);
void softmax_(Tensor&, uint32_t);
void softmax_stable_(Tensor&);
void softmax_stable_(Tensor&, uint32_t);
Tensor dropout(const Tensor&, double);
Tensor sigmoid(const Tensor&);
Tensor dsigmoid(const Tensor&, bool apply_sigmoid=true);
Tensor silu(const Tensor&);
Tensor dsilu(const Tensor&);
Tensor gelu(const Tensor&);
Tensor dgelu(const Tensor&);
Tensor tan(const Tensor&);
Tensor tanh(const Tensor&);
Tensor sin(const Tensor&);
Tensor sinh(const Tensor&);
Tensor cos(const Tensor&);
Tensor cosh(const Tensor&);
Tensor dtan(const Tensor&); // derivative of tan
Tensor dtanh(const Tensor&); // derivative of tanh
Tensor sqrt(const Tensor&);
Tensor invsqrt(const Tensor&); // 1 / sqrt(x);
Tensor dinvsqrt(const Tensor&); // derivative of invsqrt
Tensor var(const Tensor&, utils::optional_list dim = nullptr, int64_t correction = 1, bool keepdim = false); //delta degrees of freedom (0 for population variance, 1 for sample variance).
Tensor dvar(const Tensor& dx, const Tensor& x, utils::optional_list dim = nullptr, int64_t correction = 1); //derivative of the var function with respect to xi element of the the tensor
Tensor abs(const Tensor&); //absolute value
Tensor log(const Tensor&);
Tensor dlog(const Tensor&);
Tensor clamp(const Tensor& x, std::optional<int64_t> min = std::nullopt, std::optional<int64_t> max = std::nullopt);
Tensor relu(const Tensor&);
Tensor softplus(const Tensor& x, Scalar beta=1.0, Scalar threshold=20.0);
Tensor softmax(Tensor&);
Tensor softmax(Tensor&, uint32_t);
Tensor softmax_stable(Tensor&);
Tensor softmax_stable(Tensor&, uint32_t);
Tensor cat(std::vector<Tensor>);
Tensor cat(std::vector<Tensor>, int64_t dim);
Tensor cat(const Tensor&);
Tensor cat(const Tensor&, int64_t dim);
Tensor cat_unordered(const Tensor&); // a way to concatenate the tensors but not care about the shape or number of elements, the output shape can be determined by the user
Tensor cat_unordered(const std::vector<Tensor>&); // a way to concatenate the tensors but not care about the shape or number of elements, the output shape can be determined by the user
Tensor stack(std::vector<Tensor>);
Tensor stack(std::vector<std::reference_wrapper<Tensor> >);
Tensor stack(std::vector<Tensor>, int64_t dim);
Tensor stack(std::vector<std::reference_wrapper<Tensor> >, int64_t dim);
Tensor stack(const Tensor&, int64_t dim=0);
Tensor vectorize(std::vector<Tensor>);
size_t amount_of(const Tensor&, Scalar);
size_t count(const Tensor&);
Tensor where(Tensor);
Tensor index_select(Tensor input, int8_t dim, Tensor index);
Tensor select(Tensor input, int8_t dim, typename Tensor::size_value_t index);
Tensor meshgrid(Tensor&&, Tensor&&);
Tensor split(Tensor input, typename Tensor::size_value_t split_size, int64_t dim = 0); //splits into variable number of split sizes along a given dimension
Tensor split(Tensor input, std::vector<typename Tensor::size_value_t> split_sections, int64_t dim = 0); //splits into a specified amount on the given dimension
Tensor chunk(Tensor input, typename Tensor::size_value_t chunks, int64_t dim = 0); //splits into that many chunks
Tensor as_strided(const Tensor& input, const SizeRef n_size, SizeRef n_stride, const int64_t storage_offset = 0, bool whole_tensor=false);
Tensor sort(const Tensor&, const Tensor::size_value_t dim = -1, bool descending = false, bool return_sorted=true, bool return_indices=true);
//returns Tensor(sorted, indices)
//only returning the indices or the sorted makes things faster
//this function is meant to sort along for example each row in a matrix or each collumn
Tensor coordsort(const Tensor& input, Tensor::size_value_t dim = -2, bool descending = false, bool return_sorted=true, bool return_indices=true);
Tensor unique(Tensor, int64_t dim, bool return_unique=true, bool return_indices=true);
// returns Tensor(unique, indices) (depending on return_unique and return_indices)
//puts the tensors into a Tensor of dtype TensorObj with sizeof...(Args) + 1 number of tensors
Tensor combinations(Tensor vec, int64_t r, int64_t start = 0);
template<typename T, typename... Args,
         typename std::enable_if<std::is_same<std::decay_t<T>, Tensor>::value, int>::type = 0>
inline Tensor list(T&& first, Args&&... rest){
	static_assert(utils::is_all_same_v<std::decay_t<T>, std::decay_t<Args>...>, 
                  "Expected to make a list of all Tensors");
	Tensor out = Tensor::makeNullTensorArray(1 + sizeof...(Args));
	Tensor* begin = reinterpret_cast<Tensor*>(out.data_ptr());
	*begin = std::forward<T>(first);
	//assign the rest using parameter pack expansion
	Tensor* current = begin + 1; //move to the next position
	((*(current++) = std::forward<Args>(rest)), ...);
	return std::move(out);
}

template<typename T>
inline Tensor vector_to_tensor(const std::vector<T>& vec){
    static_assert(DTypeFuncs::type_is_dtype<T>, "Type in vector is unsupported");
    Tensor out({static_cast<int64_t>(vec.size())}, DTypeFuncs::type_to_dtype<T>);
    std::memcpy(out.data_ptr(), &vec[0], sizeof(T) * vec.size());
    return std::move(out);
}

}
}

namespace std{
inline ::nt::Tensor cos(const ::nt::Tensor& t) {return ::nt::functional::cos(t); }
inline ::nt::Tensor sin(const ::nt::Tensor& t) {return ::nt::functional::sin(t); }
inline ::nt::Tensor tan(const ::nt::Tensor& t) {return ::nt::functional::tan(t); }
inline ::nt::Tensor cosh(const ::nt::Tensor& t){return ::nt::functional::cosh(t);}
inline ::nt::Tensor sinh(const ::nt::Tensor& t){return ::nt::functional::sinh(t);}
inline ::nt::Tensor tanh(const ::nt::Tensor& t){return ::nt::functional::tanh(t);}
inline ::nt::Tensor sqrt(const ::nt::Tensor& t){return ::nt::functional::sqrt(t);}
inline ::nt::Tensor abs(const ::nt::Tensor& t) {return ::nt::functional::abs(t); }
inline ::nt::Tensor log(const ::nt::Tensor& t) {return ::nt::functional::log(t); }
}


#endif
