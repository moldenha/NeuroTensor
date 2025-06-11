#include "../Tensor.h"
#include "../functional/functional.h"
#include <memory>
#include <functional>
#include <string_view>
//this is going to be a wrapper for the tensor class that acts as an auto-grad engine


namespace nt{
namespace layers{
namespace autograd{

class TensorGrad;


class GradientHolder{
	std::vector<std::unique_ptr<TensorGrad> > grads;
	public:
		GradientHolder(){}
		TensorGrad* addTensor(TensorGrad* grad){
			grads.push_back(std::make_unique<TensorGrad>());
			*grads.back() = std::move(*grad);
			return grads.back().get();
		}

};




//currently there is a flaw with this, and that is if one of the TensorGrad's goes out of scope,
//could probably fix that with the module class, and hold onto a shared_ptr of the tensor grad/tensor that goes in
class GradientTracker{
	std::vector<std::unique_ptr<TensorGrad> > Tensors;
	friend class TensorGrad;
	public:
		GradientTracker() {}
		//below is bad and could easily cause a segmentation error by referencing a tensor that goes out of scope
		//so instead, I am going to see if I can make module hold onto a vector of unique_ptr<TensorGrad>
		//and have it move itself into that unique_ptr, and then reaffirm itself as what is in that ptr
		inline TensorGrad* addTensor(std::unique_ptr<TensorGrad> tensor){
			Tensors.push_back(std::move(tensor));
			return Tensors.back().get();
		}

		inline Tensor backward(Tensor grad) {
			for (auto it = Tensors.rbegin(); it != Tensors.rend(); ++it) {
				grad = it->backward(grad);
			}
			return grad;
		}
};


//this class assume's this is dX
//and with a BinaryOperator assumes the input function is dB
class TensorGrad{
	Tensor original_tensor;
	Tensor Gradient;
	std::shared_ptr<GradientTracker> tracker;
	std::shared_ptr<std::vector<function<Tensor(Tensor&)> > > operations;
	inline void add_view_change(){
		const SizeRef& sh = shape();
		trackUnaryFunction([&sh](Tensor& grad){return grad.view(sh);});
	}

	inline void generate_ranges(const std::vector<my_range>& ranges, std::vector<my_range> current_ranges, size_t idx, std::vector<std::vector<my_range>>& result, const SizeRef& shape) {
	    if (idx == ranges.size()) {
		    if(shape[idx] == ranges[idx].length()){
			    result.push_back(current_ranges);
			    return;
		    }
		    for(uint32_t i = 0; i < shape[idx]; i += ranges[idx].length()){
				if(i + ranges[idx].length() >= shape[idx]){
					current_ranges[idx] = my_range(i, -1);
					current_ranges[idx].fix(shape[idx]);
					result.push_back(current_ranges);
					break;
				}
				current_ranges[idx] = my_range(i, i + ranges[idx].length());
				result.push_back(current_ranges);
		    }
		return;
	    }
	    
	    if(shape[idx] == ranges[idx].length()){
		generate_ranges(ranges, current_ranges, idx + 1, result, shape);
	    }
	    else{
		    for (int32_t i = 0; i < shape[idx]; i += ranges[idx].length()) {
			if(i + ranges[idx].length() >= shape[idx]){
				current_ranges[idx] = my_range(i, -1);
				current_ranges[idx].fix(shape[idx]);
				generate_ranges(ranges, current_ranges, idx + 1, result, shape);
				break;
			}
			current_ranges[idx] = my_range(i, i + ranges[idx].length());
			generate_ranges(ranges, current_ranges, idx + 1, result, shape);
		}
	    }
	}
	inline void add_self(){
		std::unique_ptr<TensorGrad> this_grad = std::make_unique<TensorGrad>(*this);
		TensorGrad* raw_ptr = tracker->addTensor(std::move(this_grad));
		original_tensor = raw_ptr->original_tensor;
		Gradient = raw_ptr->Gradient;
		tracker = raw_ptr->tracker;
		operations = raw_ptr->operations;
	}
	public:
		explicit TensorGrad(const Tensor& orig) 
			:original_tensor(orig),
			operations(std::make_shared<std::vector<std::function<Tensor(Tensor&)> > >()),
			tracker(std::make_shared<GradientTracker>())
		{add_self();}
		explicit TensorGrad(Tensor&& orig) 
			: original_tensor(std::move(orig)), 
			operations(std::make_shared<std::vector<std::function<Tensor(Tensor&)> > >()),
			tracker(std::make_shared<GradientTracker>()) 
		{add_self();}
		explicit TensorGrad(Tensor&& orig, std::shared_ptr<GradientTracker> track)
			:original_tensor(orig),
			operations(std::make_shared<std::vector<std::function<Tensor(Tensor&)> > >()),
			tracker(track)
		{add_self();}
		//if operations did not get deleted, this could cause an issue
		explicit TensorGrad(const TensorGrad& tg) 
			:original_tensor(tg.original_tensor),
			operations(tg.operations),
			tracker(tg.tracker)
		{add_self();}
		explicit TensorGrad(TensorGrad&& tg)
			:original_tensor(std::move(tg.original_tensor)),
			operations(std::move(tg.operations)),
			tracker(std::move(tg.tracker))
		{add_seld();}
		//this is more of a self-gradient only thing
		//this can be used more with like view(), or *= Scalar, stuff like that
		void trackUnaryOperation(std::function<Tensor(Tensor&)> operation){ 
			// takes dX
			operations->push_back(operation);
		}
		//this is what happens when 2 tensors interact with each other
		//for example TensorGrad A += TensorGrad B;
		//A.trackBinaryOperation([](Tensor& AGrad, TensorGrad& BGrad){
		//   Tensor dB = AGrad;
		//   BGrad.Gradient += dB;
		//   return AGrad.Gradient;
		//}, B);
		//B.trackBinaryOperation([](Tensor& BGrad, TensorGrad& AGrad){
		//   Tensor dB = AGrad.Gradient;
		//   BGrad += dB;
		//   return BGrad;
		//}, A);
		//the entire idea, is that the tracker would only for example if A is returned from the module
		//only A would have backward called on it, or if B were called, only B would have backward called on it
		//However, which ever one is returned it won't matter because both will be updated
		void trackBinarOperation(std::function<Tensor(Tensor&, TensorGrad&)> operation, TensorGrad& dA) {
			// takes dX and dA
			operations->push_back([operation, dA](Tensor& tensor) {
				return operation(tensor, dA);
			});
		}


		inline Tensor& getTensor() {return original_tensor;}
		inline const Tensor& getTensor() const {return original_tensor;}
		inline TensorGrad& operator*=(Scalar s){
			original_tensor *= s;
			trackUnaryOperation([&s](Tensor& grad){return grad + (grad * s);});
			return *this;
		}
		inline TensorGrad& operator-=(Scalar s){
			original_tensor -= s;
			trackUnaryOperation([&s](Tensor& grad){return grad + (grad - s);});
			return *this;
		}

		inline TensorGrad& operator/=(Scalar s){
			original_tensor /= s;
			trackUnaryOperation([&s](Tensor& grad){return grad + (grad / s);});
			return *this;
		}
		inline TensorGrad& operator+=(Scalar s){
			original_tensor += s;
			trackUnaryOperation([&s](Tensor& grad){return grad + (grad + s);});
			return *this;

		}
		inline TensorGrad operator+(Scalar s){
			TensorGrad nT(original_tensor.clone(), tracker);
			nT += s;
		}
		inline TensorGrad operator*(Scalar s){
			TensorGrad nT(original_tensor.clone(), tracker);
			nT *= s;
		}
		inline TensorGrad operator/(Scalar s){
			TensorGrad nT(original_tensor.clone(), tracker);
			nT /= s;
		}
		inline TensorGrad operator-(Scalar s){
			TensorGrad nT(original_tensor.clone(), tracker);
			nT -= s;
		}
		inline TensorGrad& operator+=(TensorGrad& t){ //dX is this tensors gradient, Gradient is t's Gradient
			original_tensor += t.original_tensor;
			trackBinaryOperation([](Tensor& dX, TensorGrad& dT){
					dT.Gradient += dX;
					return dX;
					}, t);
			t.trackBinaryOperation([](Tensor& dT, TensorGrad& dX){
					dT += dX.Gradient;
					return dT;
					}, *this);
		}
		inline TensorGrad& operator/=(TensorGrad& t){ // need to figure this part out
			original_tensor /= t.original_tensor;
			trackBinaryOperation([](Tensor& dX, TensorGrad& dT){
					dT.Gradient /= dX;
					return dX;
					}, t);
			t.trackBinaryOperation([](Tensor& dT, TensorGrad& dX){
					dT /= dX.Gradient;
					return dT;
					}, *this);
		}
		inline TensorGrad& operator*=(TensorGrad& t){ // need to figure this part out
			original_tensor *= t.original_tensor;
			trackBinaryOperation([](Tensor& dX, TensorGrad& dT){
					dT.Gradient *= dX;
					return dX;
					}, t);
			t.trackBinaryOperation([](Tensor& dT, TensorGrad& dX){
					dT *= dX.Gradient;
					return dT;
					}, *this);
		}
		inline TensorGrad& operator-=(TensorGrad& t){ // need to figure this part out
			original_tensor -= t.original_tensor;
			trackBinaryOperation([](Tensor& dX, TensorGrad& dT){
					dT.Gradient -= dX;
					return dX;
					}, t);
			t.trackBinaryOperation([](Tensor& dT, TensorGrad& dX){
					dT -= dX.Gradient;
					return dT;
					}, *this);
		}
		inline TensorGrad operator+(TensorGrad& b){
			TensorGrad C(original_tensor.clone(), tracker);
			C += n;
			return std::move(C);
		}
		inline TensorGrad operator-(TensorGrad& b){
			TensorGrad C(original_tensor.clone(), tracker);
			C -= b;
			return std::move(C);
		}
		inline TensorGrad operator/(TensorGrad& b){
			TensorGrad C(original_tensor.clone(), tracker);
			C /= b;
			return std::move(C);
		}
		inline TensorGrad operator*(TensorGrad& b){
			TensorGrad C(original_tensor.clone(), tracker);
			C *= b;
			return std::move(C);
		}
		
		inline TensorGrad& operator++(){return *this += 1;}
		inline Tensor operator<=(const TensorGrad& tg){return original_tensor <= tg.original_tensor;}
		inline Tensor operator<=(const Tensor& tg){return original_tensor <= tg;}
		inline Tensor operator<=(Scalar tg){return original_tensor <= tg;}
		inline Tensor operator>=(const TensorGrad& tg){return original_tensor >= tg.original_tensor;}
		inline Tensor operator>=(const Tensor& tg){return original_tensor >= tg;}
		inline Tensor operator>=(Scalar tg){return original_tensor >= tg;}
		inline Tensor operator<(const TensorGrad& tg){return original_tensor < tg.original_tensor;}
		inline Tensor operator<(const Tensor& tg){return original_tensor < tg;}
		inline Tensor operator<(Scalar tg){return original_tensor < tg;}
		inline Tensor operator>(const TensorGrad& tg){return original_tensor > tg.original_tensor;}
		inline Tensor operator>(const Tensor& tg){return original_tensor > tg;}
		inline Tensor operator>(Scalar tg){return original_tensor > tg;}
		inline Tensor operator==(const TensorGrad& tg){return original_tensor == tg.original_tensor;}
		inline Tensor operator==(const Tensor& tg){return original_tensor == tg;}
		inline Tensor operator==(Scalar tg){return original_tensor == tg;}
		inline TensorGrad operator-() {return *this * -1;}
		inline const size_t dims() const {return original_tensor.dims();}
		inline const SizeRef& shape() const {return original_tensor.shape();}
		inline TensorGrad clone() const {return TensorGrad(original_tensor.clone(), tracker);}
		/* inline void trackZerosUnaryFunction */
		/* inline TensorGrad operator[](int32_t index){ */
		/* 	const SizeRef& sh = shape(); */
		/* 	const DType& dt = original_tensor.dtype; */
		/* 	trackUnaryOperation([&index, sh, dt](Tensor &grad){ */
		/* 			Tensor nGrad = ::nt::functional::zeros(sh, dt); */
		/* 			nGrad[index] = grad; */
		/* 			return std::move(nGrad); */
		/* 			}); */
		/* 	return TensorGrad(original_tensor[index], tracker); */ 
		/* } */
		inline TensorGrad operator[](const Tensor& t){
			const SizeRef& sh = shape();
			const DType& dt = original_tensor.dtype;
			trackUnaryOperation([&t, sh, dt](Tensor& grad){
					Tensor nGrad = ::nt::functional::zeros(sh, dt);
					nGrad[t] = grad;
					return std::move(nGrad);
			});	
			return TensorGrad(original_tensor[t], tracker); 
		}

		inline TensorGrad operator[](std::vector<my_range>& range){
			const SizeRef& sh = shape();
			const DType& dt = original_tensor.dtype;
			trackUnaryOperation([&range, sh, dt](Tensor& grad){
					Tensor nGrad = ::nt::functional::zeros(sh, dt);
					nGrad[range] = grad;
					return std::move(nGrad);
			});
			return TensorGrad(original_tensor[range], tracker);
		}
		
		inline TensorGrad& operator=(Scalar s){
			original_tensor = s;
			trackUnaryOperation([](Tensor& grad){grad.fill_(0);return grad;});
			return *this;
		}
		
		inline TensorGrad& _fill(Scalar s){
			return *this = s;
		}
		inline TensorGrad& _fill(const Tensor& t){
			original_tensor.fill_(t);
			trackUnaryOperation([](Tensor& grad){grad.fill_(0);return grad;});
			return *this;
		}

		inline TensorGrad& _fill(const TensorGrad& t){
			original_tensor.fill_(t.original_tensor);
			trackUnaryOperation([](Tensor& grad){grad.fill_(0); return grad;});
			return *this;
		}
		inline TensorGrad& _add(Scalar s){return *this += s;}
		inline TensorGrad& _add(const TensorGrad& s){return *this += s;}
		
		inline TensorGrad& _minus(Scalar s){return *this -= s;}
		inline TensorGrad& _minus(const TensorGrad& s){return *this -= s;}
		
		inline TensorGrad& _multiply(Scalar s){return *this *= s;}
		inline TensorGrad& _multiply(const TensorGrad& s){return *this *= s;}
		
		inline TensorGrad& _divide(Scalar s){return *this /= s;}
		inline TensorGrad& _divide(const TensorGrad& s){return *this /= s;}

		inline Scalar toScalar() const {return original_tensor.toScalar();}
		
		inline TensorGrad contiguous() const {return TensorGad(original_tensor.contiguous(), tracker);}
		
		template<typename T>
		const T& item() const {return original_tensor.item<T>();}
		template<typename T>
		T& item() {return original_tensor.item<T>();}
		
		inline const bool is_contiguous() const {return original_tensor.is_contiguous();}
		inline const uint32_t contig_count() const {return original_tensor.use_count();}
		inline void print() const {original_tensor.print();}
		inline void* data_ptr() {return original_tensor.data_ptr();}
		inline const void* data_ptr() const {return original_tensor.data_ptr();}
		inline void* data_ptr_end() {return original_tensor.data_ptr_end();}
		inline const void* data_ptr_end() const {return original_tensor.data_ptr_end();}
		inline const uint32_t& numel() const {return original_tensor.numel();}
		inline TensorGrad view(SizeRef ns){
			add_view_change();
			return TensorGrad(original_tensor.view(ns), tracker);
		}
		template<typename... Args>
		inline TensorGrad view(int64_t i, Args&&... args){
			add_view_change();
			return TensorGrad(original_tensor.view(i, args...), tracker);
		}
		
		inline TensorGrad unsqueeze(){
			add_view_change();
			return TensorGrad(original_tensor.unsqueeze(), tracker);
		}

		inline TensorGrad squeeze(){
			add_view_change();
			return TensorGrad(original_tensor.squeeze(), tracker);
		}
		
		inline friend std::ostream& operator << (std::ostream& out, const TensorGrad& tg)
		{return out << tg.original_tensor;}
		
		inline TensorGrad permute(std::vector<uint32_t> p){
			trackUnaryFunction([&p](Tensor& grad){
						return grad.permute(p);
					});
			return TensorGrad(original_tensor.permute(p), tracker);
		}
		inline TensorGrad transpose(int8_t a, int8_t b){
			trackUnaryFunction([&a, &b](Tensor& grad){return grad.transpose(a,b);});
			return TensorGrad(original_tensor.transpose(a,b), tracker);
		}
		inline TensorGrad flatten(int8_t a, int8_t b){
			add_view_change();
			return TensorGrad(original_tensor.flatten(a,b), tracker);
		}

		inline TensorGrad div(uint32_t i){
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&sh, &dt, &i](Tensor& grad){
				Tensor nGrad = ::nt::functional::zeros(sh, dt);
				nGrad.div(i) += grad;
				return std::move(nGrad);
			});
			return TensorGrad(original_tensor.div(i), tracker);
		}

		inline TensorGrad split_axis(int8_t i){
			trackUnaryFunction([](Tensor& grad){
				utils::throw_exception(grad.dtype == DType::TensorObj, "Expected to merge Tensors together from the split_axis(int8) gradient");
				return ::nt::functional::cat(grad);
			});
			add_view_change();
			return TensorGrad(original_tensor.split_axis(i), tracker);
		}

		inline TensorGrad split_axis(std::vector<my_range> ranges){
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			utils::throw_exception(ranges.size() <= dims(), "expeted to have at most $ ranges but got $ ranges for split_axis on tensor shape $", dims(), ranges.size(), shape());
			while (ranges.size() < dims()) {
				// Add a my_range(0, -1) to ranges
				ranges.push_back(my_range(0, -1));
			}
			for(uint32_t i = 0; i < ranges.size(); ++i)
				ranges[i].fix(shape()[i]);
			
			std::vector<std::vector<my_range>> result_ranges;
			std::vector<my_range> current_ranges = ranges;
			generate_ranges(ranges,current_ranges, 0, result_ranges, shape());
			if(result_ranges.size() == 1 || result_ranges.size() == 0)
				return *this;
			
			trackUnaryFunction([&sh, &dt, &result_ranges](Tensor& grad){
				utils::throw_exception(grad.dtype == DType::TensorObj, "Expected to merge Tensors together from the split_axis(vector<my_range>) gradient");
				utils::throw_exception(grad.is_contiguous(), "Expected to take gradient of contiguous tensor");
				Tensor nGrad = ::nt::functional::zeros(sh, dt);
				Tensor* tBegin = reinterpret_cast<Tensor*>(grad.data_ptr());
				for(auto begin = result_ranges.cbegin(); begin != result_ranges.cend(); ++begin, ++tBegin){
					nGrad[*begin] += *tBegin;
				}
				return std::move(nGrad);
			});
			Tensor output({static_cast<unsigned int>(result_ranges.size())}, DType::TensorObj);
			Tensor* begin = reinterpret_cast<Tensor*>(output.data_ptr());
			Tensor* end = begin + result_ranges.size();
			auto ra_begin = result_ranges.cbegin();
			for(;begin != end; ++begin, ++ra_begin)
				*begin = original_tensor[*ra_begin];
			return TensorGrad(std::move(output), tracker);
		}

		inline const ArrayVoid& arr_void() const {return original_tensor.arr();}
		inline ArrayVoid& arr_void() {return original_tensor.arr();}
		inline std::string_view sv() const {return original_tensor.sv();}
		inline TensorGrad to_dtype(const DType ndt){
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&dt](Tensor& grad){return grad.to_dtype(dt);});
			return TensorGrad(original_tensor.to_dtype(ndt), tracker);
		}

		inline TensorGrad Int(){
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&dt](Tensor& grad){return grad.to_dtype(dt);});
			return TensorGrad(original_tensor.Int(), tracker);
		}
		
		inline TensorGrad Long(){
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&dt](Tensor& grad){return grad.to_dtype(dt);});
			return TensorGrad(original_tensor.Long(), tracker);
		}

		inline TensorGrad& RowColSwap() {
			original_tensor.RowColSwap();
			trackUnaryFunction([](Tensor& grad){return grad.RowColSwap();});
		}
		
		inline TensorGrad real(){
			trackUnaryFunction([](Tensor& grad){return grad.to_complex_from_real();});
			return TensorGrad(original_tensor.real(), tracker);
		}
		inline TensorGrad imag(){
			trackUnaryFunction([](Tensor& grad){return grad.to_complex_from_imag();});
			return TensorGrad(original_tensor.imag(), tracker);
		}
		inline TensorGrad to_complex_from_real(){
			trackUnaryFunction([](Tensor& grad){return grad.real();});
			return TensorGrad(original_tensor.to_complex_from_real(), tracker);
		}
		inline TensorGrad to_complex_from_imag(){
			trackUnaryFunction([](Tensor& grad){return grad.imag();});
			return TensorGrad(original_tensor.to_complex_from_imag(), tracker);
		}

		inline TensorGrad sum(){
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&sh, &dt](Tensor& grad){
				/* Tensor nGrad = ::nt::functional::zeros(sh, dt); */
				utils::throw_exception(grad.numel() == 1, "Expected the amount of elements in grad to be 1 for backward of sum function");
				Scalar s = grad.toScalar();
				s /= sh.multiply();
				return ::nt::functional::nums(sh, s, dt); 
			});
			return TensorGrad(original_tensor.sum(), tracker);
		}
		inline TensorGrad mean(){
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&sh, &dt](Tensor& grad){
				/* Tensor nGrad = ::nt::functional::zeros(sh, dt); */
				utils::throw_exception(grad.numel() == 1, "Expected the amount of elements in grad to be 1 for backward of mean() function");
				Scalar s = grad.toScalar();
				s /= sh.multiply();
				return ::nt::functional::nums(sh, s, dt); 
			});
			return TensorGrad(original_tensor.mean(), tracker);
		}
		inline TensorGrad sum(int32_t i){
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&sh, &dt, &i](Tensor& grad){
				/* Tensor nGrad = ::nt::functional::zeros(sh, dt); */
				Tensor nGrad = ::nt::functional::zeros(sh, dt);
				Tensor split = nGrad.split_axis(i);
				grad.arr_void().execute_function<WRAP_DTYPES<RealNumberTypesL>>(
						[&split](auto a_begin, auto a_end){
							Tensor* begin = reinterpret_cast<Tensor*>(split.data_ptr());
							for(;a_begin != a_end; ++a_begin, ++begin){
								*begin = ((*a_begin) / begin->numel());
							}
						});
				return std::move(nGrad);
			});
			return TensorGrad(original_tensor.sum(i), tracker);
		}
		inline TensorGrad mean(int32_t i){
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&sh, &dt, &i](Tensor& grad){
				/* Tensor nGrad = ::nt::functional::zeros(sh, dt); */
				Tensor nGrad = ::nt::functional::zeros(sh, dt);
				Tensor split = nGrad.split_axis(i);
				grad.arr_void().execute_function<WRAP_DTYPES<RealNumberTypesL>>(
						[&split](auto a_begin, auto a_end){
							Tensor* begin = reinterpret_cast<Tensor*>(split.data_ptr());
							for(;a_begin != a_end; ++a_begin, ++begin){
								*begin = ((*a_begin) / begin->numel());
							}
						});
				return std::move(nGrad);
			});
			return TensorGrad(original_tensor.mean(i), tracker);
		}

		inline TensorGrad max(){
			size_t distance;
			Scalar outp = original_tensor.arr_void().execute_function<WRAP_DTYPES<RealNumberTypesL>>(
					[&distance](auto a_begin, auto a_end) -> Scalar{
						auto element = std::max_element(a_begin, a_end);
						distance = std::distance(a_begin, element);
						return *element;
					});
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&distance, &outp, &sh, &dt](Tensor& grad){
					utils::throw_exception(grad.numel() == 1, "Expected the amount of elements in grad to be 1 for backward of max() function but got $", grad.numel());
					
					Tensor nGrad = ::nt::functional::zeros(sh, dt);
					nGrad.flatten(0,-1);
					nGrad[distance] = grad.toScalar();
					return nGrad.view(sh);
					});
			Tensor tMax = ::nt::functional::zeros({1}, dt);
			tMax = outp;
			return TensorGrad(std::move(tMax), tracker);
		}

		inline TensorGrad max(int32_t i){
			Tensor splits = original_tensor.split_axis(i);
			std::vector<size_t> distance;
			distance.resize(splits.numel());
			std::vector<Scalar> scalars;
			scalars.resize(splits.numel());
			Tensor* begin = reinterpret_cast<Tensor*>(splits.data_ptr());
			Tensor* end = begin + splits.numel();
			auto sBegin = scalars.begin();
			auto dBegin = distance.begin();
			for(;begin != end; ++begin, ++sBegin, ++dBegin){
				*sBegin = begin->arr_void().execute_function<WRAP_DTYPES<RealNumberTypesL>>(
					[&dBegin](auto a_begin, auto a_end) -> Scalar{
						auto element = std::max_element(a_begin, a_end);
						*dBegin = std::distance(a_begin, element);
						return *element;
					});

			}
			const SizeRef& sh;
			const DType& dt = original_tensor.dtype;
			trackUnaryFunction([&distance, &scalars, &sh, &dt, &i](Tensor& grad){
					utils::throw_exception(grad.numel() == scalars.size(), "Expected the amount of elements in grad to be $ for backward of max() function but got $", scalars.size(), grad.numel());
					Tensor nGrad = ::nt::functional::zeros(sh, dt);
					grad.arr_void().execute_function<WRAP_DTYPES<RealNumberTypesL>>(
						[&nGrad, &distance](auto a_begin, auto a_end){
						ensor split = nGrad.split_axis(i);
						Tensor* begin = reinterpret_cast<Tensor*>(split.data_ptr());
						Tensor* end = begin + split.numel();
						auto dBegin = distance.begin();
						for(;begin != end; ++begin, ++dBegin){
							begin->flatten(0,-1)[*dBegin] += a_begin;
						}

						}
					)
					return std::move(nGrad);
			});
			return TensorGrad(original_tensor.max(i), tracker);

		}

		inline TensorGrad exp(){
			Tensor outp = original_tensor.exp();
			trackUnaryFunction([&outp](Tensor& grad){
				return grad + (grad * outp);
			});
			return TensorGrad(outp.clone(), tracker);
		}
		
		inline TensorGrad& exp_(){
			original_tensor.exp_();
			Tensor ex = original_tensor.clone();
			trackUnaryFunction([&original_tensor](Tensor& grad){
				return grad + (grad * orginal_tensor);
			});
			original_tensor = std::move(ex);
			return *this;
		}

		inline TensorGrad inverse(){
			Tensor orig = -original_tensor.pow(-2);
			trackUnaryFunction([orig](Tensor& grad){
				return grad += grad * orig;
			});
			return TensorGrad(original_tensor.inverse(), tracker);
		}
		inline TensorGrad& inverse_(){
			Tensor orig = -original_tensor.pow(-2);
			trackUnaryFunction([orig](Tensor& grad){
				return grad += grad * orig;
			});
			original_tensor.inverse_();
			return *this;
		}
		inline TensorGrad pow(int32_t p){
			Tensor orig = p * original_tensor.pow(p-1);
			trackUnaryFunction([orig](Tensor& grad){
				return grad += grad * orig;
			});
			return TensorGrad(original_tensor.pow(p), tracker);
		}

		inline TensorGrad clip(Scalar clip_min, Scalar clip_max) const{
			TensorGrad cpy = this->clone();
			cpy[cpy < clip_min] = clip_min;
			cpy[cpy > clip_max] = clip_max;
			return std::move(cpy);
		}

		inline TensorGrad& clip_(Scalar clip_min, Scalar clip_max){
			(*this)[cpy < clip_min] = clip_min;
			(*this)[cpy > clip_max] = clip_max;
			return *this;
		}

		inline TensorGrad pad(std::vector<uint32_t> p, const char* mode="constant", double value=0){
			utils::throw_exception(p.size() % 2 == 0, "RuntimeError: The size of the pad must have 2 per dimension");
			utils::throw_exception((p.size() / 2) <= dims(), "RuntimeError: expected padding for at most $ dims but instead got $", dims(), int(p.size() / 2));

			std::vector<uint32_t> n_shape = shape().Vec();
			uint32_t i = 0;
			uint32_t last_dims = uint32_t(p.size() / 2);
			for(; i < (p.size() / 2); ++i){
				n_shape[n_shape.size() - (i+1)] += p[i*2];
				n_shape[n_shape.size() - (i+1)] += p[i*2+1];
			}
			Tensor output(SizeRef(std::move(n_shape)), dtype);
			output = value;
			std::vector<nt::my_range> ranges(dims());
			auto begin = p.cbegin();
			uint32_t start = dims() - int32_t(p.size() / 2);
			for(i = 0; i < dims(); ++i){
				if(i < (start)){
					ranges[i] = nt::my_range(0, shape()[i]);
					continue;
				}
				ranges[i] = nt::my_range(*begin, (-1)*int32_t(*(begin + 1)));
				++begin;
				++begin;
			}
			output[ranges] = original_tensor;
			trackUnaryFunction([&ranges](Tensor& grad){
						return grad[ranges];
					});

			return TensorGrad(std::move(output), tracker);
		}

		inline TensorGrad flip(){
			trackUnaryFunction([](Tensor& grad){return grad.flip();});
			return TensorGrad(original_tensor.flip(), tracker);
		}

		inline TensorGrad flip(int32_t dim) const{
			trackUnaryFunction([&dim](Tensor& grad){return grad.flip(dim);});
			return TensorGrad(original_tensor.flip(dim), tracker);
		}

		//only ones left are dialate, I just don't feel like doing that right now

			

	
		inline std::vector<typename SizeRef::ArrayRefInt::value_type> strides() const {return original_tensor.strides();}
		
		inline bool backward_once(Tensor& grad){
			if(operations->empty()){return false;}
			grad = operations->back()(grad);
			operations->pop_back();
			return true;
		}

		inline Tensor backward(Tensor& grad){
			Gradient = grad.clone();
			while(backward_once(Gradient)){}
			return Gradient.clone();
		}

		inline void Update(Scalar lr = 0.01, bool clip=true, Scalar clip_min = -10, Scalar clip_max = 10){
			Gradient *= lr;
			if(clip){Gradient.clip_(clip_min, clip_max);}
			orignal_tensor += Gradient.
			return;
		}
		void zero_grad(){
			Gradient.fill_(0);
		}

};

}
}
}
