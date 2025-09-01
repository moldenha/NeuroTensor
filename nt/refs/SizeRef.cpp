#include "SizeRef.h"
#include "ArrayRef.h"


#include <cstddef>
#include <vector>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <assert.h>

namespace nt{

/* SizeRef::SizeRef(ArrayRefInt Arr) */
/* 	:_sizes(Arr.size() + 1), _mults(Arr.size() + 1) */
/* { */
/* 	_sizes[0] = Arr; */
/* 	_mults[0] = Arr.multiply(); */
/* 	uint16_t i = 0; */
/* 	while(!_sizes.at(i).empty()){ */
/* 		_sizes.at(i+1) = _sizes.at(i).pop_front(); */
/* 		_mults.at(i+1) = _sizes.at(i+1).multiply(); */
/* 		++i; */
/* 	} */
/* 	_mults.back() = 1; */
/* } */
// SizeRef check_1({30});

// void size_ref_check(const SizeRef& arr){
//     if(check_1 == arr){std::cout << "size ref of "<<check_1<<" constructed! = "<<arr<<std::endl;return;}
// }

SizeRef::SizeRef(const SizeRef& Arr)
	:_sizes(Arr._sizes)
{
	// // size_ref_check(*this);
    //std::cout<<"copying size ref"<<std::endl;
}



SizeRef::SizeRef(SizeRef&& Arr)
	:_sizes(std::move(Arr._sizes))
{
	// size_ref_check(*this);

}

SizeRef::SizeRef(const ArrayRefInt& Arr)
	:_sizes(Arr)
{
	// size_ref_check(*this);

}

SizeRef::SizeRef(ArrayRefInt&& Arr)
	:_sizes(std::move(Arr))
{
	// size_ref_check(*this);

}

SizeRef& SizeRef::operator=(const SizeRef &Arr){
	_sizes = Arr._sizes;
	// size_ref_check(*this);
	return *this;

}

SizeRef& SizeRef::operator=(SizeRef&& Arr){
	_sizes = std::move(Arr._sizes);
	// size_ref_check(*this);
	return *this;
}

SizeRef::SizeRef(const value_type &OneEle)
	:_sizes({OneEle})
{
	// size_ref_check(*this);

}

SizeRef::SizeRef(const ArrayRefInt::value_type *data, size_t length)
	:_sizes(data, length)
{
	// size_ref_check(*this);

}

SizeRef::SizeRef(const std::vector<ArrayRefInt::value_type> &Vec)
	:_sizes(&Vec[0], Vec.size())
{
	// size_ref_check(*this);

}

SizeRef::SizeRef(std::vector<ArrayRefInt::value_type>&& Vec)
	:_sizes(&Vec[0], Vec.size())
{
	// size_ref_check(*this);

}


template<size_t N>
SizeRef::SizeRef(const std::array<value_type, N> &Arr)
	:_sizes(Arr)
{
	// size_ref_check(*this);

}


template<size_t N>
SizeRef::SizeRef(const value_type (&Arr)[N])
	:_sizes(Arr)
{
	// size_ref_check(*this);

}

// SizeRef::SizeRef(const std::initializer_list<ArrayRefInt::value_type> &Vec)
// 	:_sizes(Vec)
// {
// 	// size_ref_check(*this);

// }

SizeRef::SizeRef(std::nullptr_t)
	:_sizes(nullptr)
{}

bool SizeRef::operator==(const SizeRef &Arr) const {return _sizes == Arr._sizes;}
bool SizeRef::operator!=(const SizeRef &Arr) const {return _sizes != Arr._sizes;}
const typename SizeRef::value_type* SizeRef::data() const {return _sizes.data();}
typename SizeRef::value_type SizeRef::size() const {return static_cast<typename SizeRef::value_type>(_sizes.size());}
const typename SizeRef::value_type& SizeRef::front() const {return _sizes.front();}
const typename SizeRef::value_type& SizeRef::back() const {return _sizes.back();}
typename SizeRef::ArrayRefInt::iterator SizeRef::begin() const {return _sizes.begin();}
typename SizeRef::ArrayRefInt::iterator SizeRef::end() const {return _sizes.end();}
typename SizeRef::ArrayRefInt::const_iterator SizeRef::cbegin() const {return _sizes.cbegin();}
typename SizeRef::ArrayRefInt::const_iterator SizeRef::cend() const {return _sizes.cend();}
typename SizeRef::ArrayRefInt::reverse_iterator SizeRef::rbegin() const {return _sizes.rbegin();}
typename SizeRef::ArrayRefInt::reverse_iterator SizeRef::rend() const {return _sizes.rend();}
bool SizeRef::empty() const {return (_sizes.empty() || front() == 0);}
const typename SizeRef::ArrayRefInt::value_type& SizeRef::operator[](value_type x) const {
	x = x < 0 ? x + size() : x;
	return _sizes[x];
}

SizeRef SizeRef::operator[](range_ r) const{
	/* std::cout <<"range: "<< r << std::endl; */
	r.fix(size());
	/* std::cout <<"range: "<< r << std::endl; */
	std::vector<value_type> outp(cbegin() + r.begin, cbegin() + r.end);
	return SizeRef(outp);
}

// Look into changing this function to make it be size() + (end+1) to make -1 a valid range
// However, this breaks the linear function
// So change it, correct it, and make sure that the linear function then works

SizeRef SizeRef::range(value_type begin, value_type end) const{
	begin = begin < 0 ? size() + begin : begin;
	end = end < 0 ? size() + end : end;
	if(begin > end){std::swap(begin, end);}
	return SizeRef(std::vector<value_type>(cbegin() + begin, cbegin() + end));
}

/* typename SizeRef::ArrayRefInt::value_type& SizeRef::operator[](value_type x){ */
/* 	x = x < 0 ? x + size() : x; */
/* 	return _sizes[0][x]; */
/* } */
typename SizeRef::value_type SizeRef::multiply(value_type i) const {
	return std::accumulate<typename SizeRef::ArrayRefInt::const_iterator, value_type>(_sizes.cbegin() + (i < 0 ? i + size() : i), _sizes.cend(), 1, std::multiplies<typename ArrayRefInt::value_type>());
}

typename SizeRef::value_type SizeRef::unsigned_multiply(value_type i) const {
	auto begin = cbegin();
	auto end = cend();
	for(;begin != end; ++begin){utils::throw_exception(*begin > 0, "Expected dimensions to all be greater than 0, but got size of $", *this);}
	return std::accumulate<typename SizeRef::ArrayRefInt::const_iterator, value_type>(_sizes.cbegin() + (i < 0 ? i + size() : i), _sizes.cend(), 1, std::multiplies<typename ArrayRefInt::value_type>());
}

SizeRef SizeRef::permute(const std::vector<value_type> &Vec) const {
	auto outp = this->Vec();
    value_type max_size = static_cast<value_type>(_sizes.size());
	for(value_type i = 0; i < max_size; ++i){
		outp[i] = _sizes[Vec[i]];
	}
	return SizeRef(std::move(outp));
}
SizeRef SizeRef::transpose(value_type _a, value_type _b) const{
	_a = _a < 0 ? size() + _a : _a;
	_b = _b < 0 ? size() + _b : _b;
	auto vec = Vec();
	std::swap(vec[_a], vec[_b]);
	return SizeRef(std::move(vec));
}

SizeRef SizeRef::redo_index(value_type index, value_type val) const{
	index = index < 0 ? size() + index : index;
	auto vec = Vec();
	vec[index] = val;
	return SizeRef(std::move(vec));
}

SizeRef SizeRef::delete_index(value_type index) const{
	index = index < 0 ? size() + index : index;
	std::vector<value_type> o(size()-1);
	for(value_type i = 0; i < index; ++i){
		o[i] = _sizes[i];
	}
	for(value_type i = index + 1; i < size(); ++i){
		o[i-1] = _sizes[i];
	}
	return SizeRef(std::move(o));
}

/* const typename SizeRef::ArrayRefInt::value_type SizeRef::permute_to_index(const std::vector<ArrayRefInt::value_type> &Vec, const std::vector<ArrayRefInt::value_type> &Perms, const SizeRef& _s) const{ */
/* 	typename ArrayRefInt::value_type mult =	0; */
/* 	for(uvalue_type i = 0; i < Vec.size() - 1; ++i) */
/* 		mult += _s._mults[i+1] * Vec[Perms[i]]; */
/* 	return mult + Vec[Perms.back()]; */
/* } */
/* const typename SizeRef::ArrayRefInt::value_type SizeRef::to_index(const std::vector<ArrayRefInt::value_type> &Vec) const{ */
/* 	typename ArrayRefInt::value_type mult = std::inner_product(Vec.cbegin(), Vec.cend(), cbegin(), 0.0); */
/* 	return mult + Vec.back(); */
/* } */
/* std::pair<const std::vector<typename SizeRef::ArrayRefInt::value_type>, SizeRef> SizeRef::get_index_order(const std::vector<ArrayRefInt::value_type> &Perm) const{ */
/* 	SizeRef n_size(this->permute(Perm)); */

/* 	std::vector<typename ArrayRefInt::value_type> Vec(begin(), end()); */
	
/* 	std::for_each(Vec.begin(), Vec.end(), [](auto& val){val -= 1;}); */
/* 	std::vector<typename ArrayRefInt::value_type> example(Vec.size(), 0); */
/* 	std::vector<typename ArrayRefInt::value_type> output(_mults[0]); */
/* 	value_type index = 0; */
/* 	while(example != Vec){ */
/* 		for(value_type i = 0; i <=Vec.back(); ++i){ */
/* 			example.back() = i; */
/* 			output[index] = this->permute_to_index(example, Perm, n_size); */
/* 			++index; */
/* 		} */
/* 		value_type cur_index = Vec.size() -2; */
/* 		example[cur_index] += 1; */
/* 		while(true){ */
/* 			if(cur_index == 0) */
/* 				break; */
/* 			if(example[cur_index] > Vec[cur_index]){ */
/* 				example[cur_index] = 0; */
/* 				cur_index -= 1; */
/* 				example[cur_index] += 1; */
/* 				continue; */
/* 			} */
/* 			break; */
/* 		} */
/* 	} */
/* 	for(value_type i = 0; i <=Vec.back(); ++i){ */
/* 		example.back() = i; */
/* 		output[index] = this->permute_to_index(example, Perm, n_size); */
/* 		++index; */
/* 	} */
/* 	return std::make_pair(std::move(output), std::move(n_size)); */
/* } */

std::vector<SizeRef::ArrayRefInt::value_type> SizeRef::strides() const {
	std::vector<value_type> outp(size() + 1);
	for(value_type i = 0; i < size(); ++i){
		outp[i] = multiply(i);
	}
	outp.back() = 1;
	return std::move(outp);
}

SizeRef SizeRef::pop_front() const{
	if(size() == 1 && _sizes[0] > 1){
		return SizeRef(ArrayRefInt(1));
	}
	else if(size() == 1 && _sizes[0] == 1){
		return SizeRef(ArrayRefInt(0));
	}
	if(empty()){
		return *this;
	}
	return SizeRef(_sizes.cbegin() + 1, size()-1);
}

SizeRef SizeRef::flatten(value_type _a, value_type _b) const{
	_a = _a < 0 ? _a + size() : _a;
	_b = _b < 0 ? _b + size() : _b;
	value_type begin = _a < _b ? _a : _b;
	value_type end = _a < _b ? _b : _a;
	++end;
	typedef typename ArrayRefInt::value_type value_t;
	value_type n_dims = size() - (end - begin) + 1;
	ArrayRefInt n_vals = ArrayRefInt::zeros(n_dims); 
	std::copy(cbegin(), cbegin() + begin, n_vals.d_data());
	n_vals[begin] = std::accumulate<typename SizeRef::ArrayRefInt::const_iterator, value_type>(cbegin() + begin, cbegin() + end, value_t(1), std::multiplies<value_t>());
	std::copy(cbegin() + end, cend(), n_vals.d_data() + begin + 1);
	return SizeRef(std::move(n_vals));
}

SizeRef SizeRef::unflatten(value_type _a, value_type _b) const{
	_a = _a < 0 ? _a + size() : _a;
	_b = _b < 0 ? _b + size() : _b;
	value_type begin = _a < _b ? _a : _b;
	value_type end = _a < _b ? _b : _a;
	/* std::cout << "Begin: "<<(int)begin << " End: "<<(int)end<<std::endl; */
	value_type n_dims = size() + (end - begin);
	/* std::cout << "N_Dims: "<<n_dims<<std::endl; */
	/* std::cout<<"n_dims is "<<n_dims<<std::endl; */
	ArrayRefInt n_vals = ArrayRefInt::zeros(n_dims); 
	/* for(value_type i = 0; i < begin; ++i) */
	/* 	n_vals[i] = (*this)[i]; */
	/* for(value_type i = begin; i < end; ++i) */
	/* 	n_vals[i] = 1; */
	/* for(value_type i = end; i < n_vals.size(); ++i) */
	/* std::cout<<"made n_vals"<<std::endl; */	
	std::copy(cbegin(), cbegin() + begin, n_vals.d_data());
	std::for_each(n_vals.d_data() + begin, n_vals.d_data() + end, [](auto& v){v = 1;});
	std::copy(cbegin() + begin, cend(), n_vals.d_data() + end);
	return SizeRef(std::move(n_vals));
}

std::vector<typename SizeRef::value_type> SizeRef::Vec() const{
	return std::vector<value_type>(cbegin(), cend());
}

std::ostream& operator<< (std::ostream &out, const SizeRef& obj) {
	out << '{';
     SizeRef::value_type max_size = static_cast<SizeRef::value_type>(obj._sizes.size()) - 1;
	for(SizeRef::value_type i = 0; i < max_size; ++i)
		out << obj._sizes[i] << ',';
	out << obj.back() << '}';
	return out;
}

}
