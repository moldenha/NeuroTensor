#ifndef __NT_FUNCTIONAL_TENSOR_FILES_COMBINE_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_COMBINE_H__

#include "../../Tensor.h"
#include <vector>
#include <functional>

namespace nt {
namespace functional {

Tensor cat(const Tensor& _a, const Tensor &_b, int64_t dim=0);
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

} // namespace functional
} // namespace nt

#endif
