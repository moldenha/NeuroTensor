

//this stores the old matrix multiplication stuff:

#include <immintrin.h>

#ifdef USE_PARALLEL

template<typename T>
inline void mat_mult_single(const int& segmentationIndex, T* o_begin, const uint32_t& iterationsPerSegment, const uint32_t* numSegments, const uint32_t& k_total, const uint32_t& cols, tdtype_list<const T> a_begin, tdtype_list<const T> b_begin){
	uint32_t y = 0;
	uint32_t k_counter = 0;
	T* begin = o_begin + (segmentationIndex * iterationsPerSegment);
	T* end = o_begin + (segmentIndex == numSegments - 1) ? totalIterations : start + iterationsPerSegment;
	for(;begin != end; ++begin, ++y){
		k_counter = 0;
		for(uint32_t k = 0; k < k_total; ++k, k_counter += cols)
			*begin += *(a_begin + k) * *(b_begin + (k_counter + y));
	}
}

inline static constexpr auto mult_matrix_dims = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s, const SizeRef& b_s, DType _dt) -> Tensor{
	utils::throw_exception(a_s[-1] == b_s[-2],"\nRuntimeError: Expected second tensor rows to be $ when it was $ c",a_s[-1],b_s[-2]);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& k_total = a_s[1];
	const size_value_t& cols = b_s[1];
	Tensor output = zeros({a_s[0], cols}, _dt);
	const uint32_t total_iterations = output.numel();
	const uint32_t numSegments = a_s[0];
	const uint32_t iterationsPerSegment = totalIterations / numSegments;
	value_t* o_begin = reinterpret_cast<value_t*>(output.data_ptr());
	tbb::parallel_for(0, numSegments, [&](int segmentationIndex){
		mat_mult_single(segmentationIndex, o_begin, iterationsPerSegment, numSegments, k_total, cols, a_begin, b_begin);
	});
	return std::move(output);
};


inline static constexpr auto mult_tensor_dims = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s_o, const SizeRef& b_s_o, DType _dt) -> Tensor{
	SizeRef a_s = a_s_o.flatten(0, -3);
	SizeRef b_s = b_s_o.flatten(0, -3);
	utils::throw_exception(a_s[-1] == b_s[-2],
			"\nRuntimeError: Expected second tensor rows to be $ when it was $ b",a_s[-1],b_s[-2]);
	utils::throw_exception(a_s[0] == b_s[0], "\nmust have same out dim number before matrix dim num, expected $ got $", a_s, b_s);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& cols = b_s[2];
	const size_value_t& k_total = a_s[2];
	Tensor output = zeros({a_s[0], a_s[1], cols}, _dt);
	value_t* o_begin = reinterpret_cast<value_t*>(output.data_ptr());
	
	//inner itteration 
	const uint32_t total_iterations = a_s[1] * cols;
	const uint32_t numSegments = a_s[1];
	const uint32_t iterationsPerSegment = totalIterations / numSegments;

	const uint32_t& iterationsA = a_s.multiply(1);
	const uint32_t& iterationsB = b_s.multiply(1);
	const uint32_t& iterationsO = output.shape().multiply(1);

	//outter itterations
	const uint32_t total_outter_iterations = a_s[0];
	tbb::parallel_for(0, total_outter_iterations, [&](int outter_segmentIndex){
	auto ba_begin = a_begin + (outter_segmentIndex * iterationsA);
	auto bb_begin = b_begin + (outter_segmentIndex * iterationsB);
	value_t* begin = o_begin + (outter_segmentIndex * iterationsO);
	tbb::parallel_for(0, total_iterationsm [&](int segmentIndex){mat_mult_single(segmentIndex, begin, iterationsPerSegment, numSegments, k_total, cols, ba_begin, bb_begin);});});
};

inline static constexpr auto mult_tensor_dims_ne = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s_o, const SizeRef& b_s_o, DType _dt) -> Tensor{
	SizeRef a_s = a_s_o.flatten(0, a_s_o.size() - b_s_o.size() - 1).flatten(1,-3);
	SizeRef b_s = b_s_o.flatten(0, -3);
	utils::throw_exception(a_s[-1] == b_s[-2],
			"\nRuntimeError: Expected second tensor rows to be $ when it was $ a",a_s[-1],b_s[-2]);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& _z = a_s[1];
	const size_value_t& cols = b_s[2];
	Tensor output = zeros({a_s[0], _z, a_s[2], cols}, _dt);
	value_t* o_begin = reinterpret_cast<value_t*>(output.data_ptr());
	const size_value_t col_minus = cols - 1;
	const size_value_t& k_total = a_s[3];
	
	//inner itteration 
	const uint32_t total_iterations = a_s[1] * cols;
	const uint32_t numSegments = a_s[1];
	const uint32_t iterationsPerSegment = totalIterations / numSegments;

	const uint32_t& iterationsA1 = a_s.multiply(1);
	const uint32_t& iterationsA2 = a_s.multiply(2);
	const uint32_t& iterationsB = b_s.multiply(1);
	const uint32_t& iterationsO1 = output.shape().multiply(1);
	const uint32_t& iterationsO2 = output.shape().multiply(2);
	
	const uint32_t total_outter_iterations = a_s[1];
	tbb::parallel_for(0, a_s[0], [&, b_begin](int first_segmentIndex){
	auto ba_begin_1 = a_begin + (first_segmentIndex * iterationsA1);
	value_t* begin2 = o_begin + (first_segmentIndex * iterationsO1);  
	tbb::parallel_for(0, total_outter_iterations, [&](int outter_segmentIndex){
	auto ba_begin = ba_begin_1 + (outter_segmentIndex * iterationsA2);
	auto bb_begin = b_begin + (outter_segmentIndex * iterationsB);
	value_t* begin = begin2 + (outter_segmentIndex * iterationsO2);
	tbb::parallel_for(0, total_iterationsm [&](int segmentIndex){mat_mult_single(segmentIndex, begin, iterationsPerSegment, numSegments, k_total, cols, ba_begin, bb_begin);});});
	});
	return std::move(output);
};

#else


//make custom iterator so that std::accumulate can be used
inline static constexpr auto mult_matrix_dims = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s, const SizeRef& b_s, DType _dt) -> Tensor{
	utils::throw_exception(a_s[-1] == b_s[-2],"\nRuntimeError: Expected second tensor rows to be $ when it was $",a_s[-1],b_s[-2]);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& cols = b_s[1];
	const size_value_t col_minus = cols - 1;
	const size_value_t& k_total = a_s[1];
	Tensor output = zeros({a_s[0], cols}, _dt);
	value_t* begin = reinterpret_cast<value_t*>(output.data_ptr());
	value_t* end = begin + output.numel();
	uint32_t y = 0;
	uint32_t k_counter = 0;
	for(;begin != end; ++begin){
		k_counter = 0;
		for(uint32_t k = 0; k < k_total; ++k, k_counter += cols)
			*begin += *(a_begin + k) * *(b_begin + (k_counter + y));
		if(y == col_minus){y = 0; a_begin += k_total;} else {++y;}
	}
	return std::move(output);
};

inline static constexpr auto mult_tensor_dims = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s_o, const SizeRef& b_s_o, DType _dt) -> Tensor{
	//flatten:
	SizeRef a_s = a_s_o.flatten(0, -3);
	SizeRef b_s = b_s_o.flatten(0, -3);
	utils::throw_exception(a_s[-1] == b_s[-2],
			"\nRuntimeError: Expected second tensor rows to be $ when it was $",a_s[-1],b_s[-2]);
	utils::throw_exception(a_s[0] == b_s[0], "\nmust have same out dim number before matrix dim num, expected $ got $", a_s, b_s);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& cols = b_s[2];
	const size_value_t col_minus = cols - 1;
	const size_value_t& k_total = a_s[2];
	const size_value_t& b_total = b_s.multiply(1);
	Tensor output = zeros({a_s[0], a_s[1], cols}, _dt);
	const size_value_t& o_total = output.shape().multiply(1);	
	uint32_t k_counter = 0;
	value_t* begin = reinterpret_cast<value_t*>(output.data_ptr());
	value_t* end = begin + o_total;
	uint32_t y = 0;
	for(uint32_t z = 0; z < a_s[0]; ++z){
		for(;begin != end; ++begin){
			k_counter = 0;
			for(uint32_t k = 0; k < k_total; ++k, k_counter += cols)
				*begin += *(a_begin + k) * *(b_begin + (k_counter + y));
			if(y == col_minus){y = 0; a_begin += k_total;} else {++y;}

		}
		b_begin += b_total;
		end += o_total;
	}
	std::vector<size_value_t> resize = a_s_o.Vec();
	resize.back() = b_s[2];
	output = output.view(resize);	
	return std::move(output);
};

inline static constexpr auto mult_tensor_dims_ne = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s_o, const SizeRef& b_s_o, DType _dt) -> Tensor{
	SizeRef a_s = a_s_o.flatten(0, a_s_o.size() - b_s_o.size() - 1).flatten(1,-3);
	SizeRef b_s = b_s_o.flatten(0, -3);
	utils::throw_exception(a_s[-1] == b_s[-2],
			"\nRuntimeError: Expected second tensor rows to be $ when it was $",a_s[-1],b_s[-2]);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& _z = a_s[1];
	const size_value_t& cols = b_s[2];
	Tensor output = zeros({a_s[0], _z, a_s[2], cols}, _dt);
	value_t* o_begin = reinterpret_cast<value_t*>(output.data_ptr());
	const size_value_t col_minus = cols - 1;
	const size_value_t& k_total = a_s[3];
	const size_value_t& b_total = b_s.multiply(1);
	const size_value_t& o_total = output.shape().multiply(2);
	auto b_begin_copy = b_begin;
	value_t* o_end = o_begin + o_total;
	uint32_t k_counter = 0;
	uint32_t y = 0;
	for(uint32_t i = 0; i < a_s[0]; ++i){
		for(uint32_t z = 0; z < a_s[0]; ++z){
			for(;o_begin != o_end; ++o_begin){
				k_counter = 0;
				for(uint32_t k = 0; k < k_total; ++k, k_counter += cols)
					*o_begin += *(a_begin + k) * *(b_begin + (k_counter + y));
				if(y == col_minus){y = 0; a_begin += k_total;} else {++y;}
			}
			b_begin += b_total;
		}
		b_begin = b_begin_copy;
	}
	return std::move(output);

};


//this is when the input first tensor has a dimension of 2 and the other does not
inline static constexpr auto mult_tensor_dims_2_ne = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s_o, const SizeRef& b_s_o, DType _dt) -> Tensor{
	SizeRef a_s = a_s_o; // dim = 2
	SizeRef b_s = b_s_o.flatten(0, -3);
	utils::throw_exception(a_s[-1] == b_s[-2],
			"\nRuntimeError: Expected second tensor rows to be $ when it was $",a_s[-1],b_s[-2]);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& cols = b_s[-1];
	Tensor output = zeros({b_s[0],a_s[0], cols}, _dt);
	const size_value_t col_minus = cols - 1;
	const size_value_t& k_total = a_s[1];
	const size_value_t& b_total = b_s.multiply(1);
	const size_value_t& o_total = output.shape().multiply(1);	
	value_t* o_begin = reinterpret_cast<value_t*>(output.data_ptr());
	value_t* o_end = o_begin + o_total;
	uint32_t k_counter = 0;
	uint32_t y = 0;
	for(uint32_t i = 0; i < b_s[0]; ++i){
		auto a_cpy = a_begin;
		for(;o_begin != o_end; ++o_begin){
			k_counter = 0;
			for(uint32_t k = 0; k < k_total; ++k, k_counter += cols)
				*o_begin += *(a_begin + k) * *(b_begin + (k_counter + y));
			if(y == col_minus){y = 0; a_begin += k_total;} else {++y;}
		}
		o_end += o_total;
		a_begin = a_cpy;
		b_begin += b_total;
	}

	std::vector<uint32_t> output_shape = b_s_o.Vec();
	output_shape[output_shape.size()-2] = a_s[0];
	return output.view(SizeRef(output_shape));
};


//this is when the input first tensor has a dimension of (2 >) and the other == 2
inline static constexpr auto mult_tensor_dims_ne_2 = [](auto a_begin, auto a_end, auto b_begin, const SizeRef& a_s_o, const SizeRef& b_s_o, DType _dt) -> Tensor{
	SizeRef a_s = a_s_o.flatten(0,-3); // dim = 2
	SizeRef b_s = b_s_o; // dim = 2
	utils::throw_exception(a_s[-1] == b_s[-2],
			"\nRuntimeError: Expected second tensor rows to be $ when it was $",a_s[-1],b_s[-2]);
	typedef typename SizeRef::ArrayRefInt::value_type size_value_t;
	using value_t = typename std::remove_const<typename decltype(a_begin)::value_type>::type;
	const size_value_t& cols = b_s[1];
	Tensor output = zeros({a_s[0],a_s[1], cols}, _dt);
	const size_value_t col_minus = cols - 1;
	const size_value_t& k_total = a_s[2];
	const size_value_t& b_total = b_s.multiply(1);
	const size_value_t& o_total = output.shape().multiply(1);	
	value_t* o_begin = reinterpret_cast<value_t*>(output.data_ptr());
	value_t* o_end = o_begin + o_total;
	uint32_t k_counter = 0;
	uint32_t y = 0;
	for(uint32_t i = 0; i < a_s[0]; ++i){
		for(;o_begin != o_end; ++o_begin){
			k_counter = 0;
			for(uint32_t k = 0; k < k_total; ++k, k_counter += cols)
				*o_begin += *(a_begin + k) * *(b_begin + (k_counter + y));
			if(y == col_minus){y = 0; a_begin += k_total;} else {++y;}
		}
		o_end += o_total;
	}

	std::vector<uint32_t> output_shape = a_s_o.Vec();
	output_shape.back() = cols;
	return output.view(SizeRef(output_shape));
};

#endif

//implement new version that uses the split tensor function
//it will make this faster, and more memory efficient
//well, same in terms of memory efficiency
//but then there would be at most 1 (or 2 including the use parallel) function(s) to improve upon and optimize
//and only at most 2 outside loops
Tensor matmult(const Tensor& a, const Tensor& b){
	utils::throw_exception(a.dtype == b.dtype, "\nRuntimeError: Expected second tensor to have dtype of $, instead had dtype of $", a.dtype, b.dtype);
	utils::throw_exception(a.dims() > 1 && b.dims() > 1, "\nRuntimeError: Expected tensors to have dims greater than 1, but instead had dims of $ and $", a.dims(), b.dims());
	if(a.dims() != b.dims()){
		if(a.dims() == 2){
			return a.arr_void().cexecute_function_nbool(mult_tensor_dims_2_ne, b.arr_void(), a.shape(), b.shape(), a.dtype);
		}
		if(b.dims() == 2){
			return a.arr_void().cexecute_function_nbool(mult_tensor_dims_ne_2, b.arr_void(), a.shape(), b.shape(), a.dtype);
		}
		return a.arr_void().cexecute_function_nbool(mult_tensor_dims_ne, b.arr_void(), a.shape(), b.shape(), a.dtype);
	}
	if(a.dims() == 2)
		return a.arr_void().cexecute_function_nbool(mult_matrix_dims, b.arr_void(), a.shape(), b.shape(), a.dtype);
	return a.arr_void().cexecute_function_nbool(mult_tensor_dims, b.arr_void(), a.shape(), b.shape(), a.dtype);
}


float dotProductSIMD_float(const float* a, const float* b, int size) {
    int remainder = size % 8;
    __m256 sum = _mm256_setzero_ps();

    for (int i = 0; i < size - remainder; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    // Sum the remaining elements
    float result = 0.0f;
    float* sum_arr = reinterpret_cast<float*>(&sum);
    for (int i = 0; i < 8; ++i) {
        result += sum_arr[i];
    }

    // Process the remainder
    for (int i = size - remainder; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double dotProductSIMD_double(const double* a, const double* b, int size) {
    int remainder = size % 4;
    __m256d sum = _mm256_setzero_pd();

    for (int i = 0; i < size - remainder; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(&a[i]);
        __m256d b_vec = _mm256_loadu_pd(&b[i]);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(a_vec, b_vec));
    }

    // Sum the remaining elements
    double result = 0.0;
    double* sum_arr = reinterpret_cast<double*>(&sum);
    for (int i = 0; i < 4; ++i) {
        result += sum_arr[i];
    }

    // Process the remainder
    for (int i = size - remainder; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

float16_t dot_product(float16_t* a, float16_t* b, size_t size) {
    __m512i sum = _mm512_setzero_si512();
    for (size_t i = 0; i < size; i += 16) {
        __m512i a_data = _mm512_load_si512((__m512i*)&a[i]);
        __m512i b_data = _mm512_load_si512((__m512i*)&b[i]);
        sum = _mm512_add_epi16(sum, _mm512_mulhi_epi16(a_data, b_data));
    }

    // Sum all the 16-bit values in the result
    __m256i sum_256 = _mm512_extracti64x4_epi64(sum, 1);
    sum_256 = _mm256_add_epi16(sum_256, _mm512_castsi512_si256(sum));
    __m128i sum_128 = _mm256_extracti128_si256(sum_256, 1);
    sum_128 = _mm_add_epi16(sum_128, _mm256_castsi256_si128(sum_256));

    // Sum the 8 values in the final 128-bit integer
    int result = _mm_extract_epi16(sum_128, 0) +
                 _mm_extract_epi16(sum_128, 1) +
                 _mm_extract_epi16(sum_128, 2) +
                 _mm_extract_epi16(sum_128, 3) +
                 _mm_extract_epi16(sum_128, 4) +
                 _mm_extract_epi16(sum_128, 5) +
                 _mm_extract_epi16(sum_128, 6) +
                 _mm_extract_epi16(sum_128, 7);

    // Convert the result to a float if necessary
    return (float16_t)result;
}

//this will be done for if the tensors are floats, and if a is contiguous, b is transposed and turned contiguous. 
Tensor mat_mult_floats(const Tensor& a, const Tensor& b){
	
}



Tensor new_matmult(const Tensor& a, const Tensor& b, bool un_transpose){
	utils::throw_exception(a.dtype == b.dtype, "\nRuntimeError: Expected second tensor to have dtype of $, instead had dtype of $", a.dtype, b.dtype);
	utils::throw_exception(a.dims() > 1 && b.dims() > 1, "\nRuntimeError: Expected tensors to have dims greater than 1, but instead had dims of $ and $", a.dims(), b.dims());
	utils::throw_exception(a_s[-1] == b_s[-2], "\nRuntimeError: Expected second tensor rows to be $ when it was $",a_s[-1],b_s[-2]);
	if(a.dims() != b.dims()){
		if(a.dims() > b.dims()){
			std::vector<uint32_t> size_outp = a.shape().Vec();
			size_outp.back() = b.shape().back();
			uint32_t start = a.dims() - b.dims() - 1;
			for(uint32_t i = start; i < size_outp.size()-2; ++i){
				utils::throw_exception(size_outp[i] == b.shape()[i - start], "RuntimeError: Expected sizes at dimension $ to be $ but got $", i-start, size_outp[i], b.shape()[i-start]); 
			}
			Tensor output = zeros(SizeRef(std::move(size_outp)), a.dtype);
			/* make some thing like:
			 * auto outp_begin = output.val_begin() (something like this
			 */
			const Tensor a_1 = a.split_axis(start);
			const Tensor* a1_begin = reinterpret_cast<const Tensor*>(a_1.data_ptr());
			const Tensor* a1_end = a1_begin + a_1.numel();
			b.RowColSwap();
			const Tensor b1 = b.split_axis(-2);
			for(;a1_begin != a1_end; ++a1_begin){
				const Tensor* b1_begin = reinterpret_cast<const Tensor*>(b1.data_ptr());
				const Tensor* b1_end = b1_begin + b1.numel();
				const Tensor a2 = a1_begin->split_axis(-2);
				const Tensor* a2_begin = reinterpret_cast<const Tensor*>(a2.data_ptr());
				const Tensor* a2_end = a2_begin + a2.numel();
				for(;b1_begin != b1_end; ++b1_begin){ // also itterate the output_iterator
				 //then do dot product of b1_begin and a2_begin and set that outp_begin
				}

			}


		}
	}
}
