#include "../Tensor.h"
#include "../utils/utils.h"
#include "../functional/functional.h"
#include <_types/_uint8_t.h>
#include <array>
#include <sys/_types/_int64_t.h>
#include "Points.h"
#include "../dtype/ArrayVoid.hpp"
#include <map>
#include <algorithm>

namespace nt{
namespace tda{

template<size_t N>
Points<N>::Points(const Tensor& t, const uint8_t point)
	:original(t.to_dtype(DType::uint8)), coords(functional::where(t == point))
{
	utils::throw_exception(t.dims() == N, "\nRuntimeError: Expected tensor to have $ dims but got $", N, t.dims());

}

template<size_t N>
Points<N>::Points(Tensor&& t, const uint8_t point)
	:original(std::move(t))
{
	std::cout<<"getting coords"<<std::endl;
	coords = std::move(functional::where(original == point));
	std::cout<<"got coords"<<std::endl;
	utils::throw_exception(original.dims() == N, "\nRuntimeError: Expected tensor to have $ dims but got $", N, original.dims());

}



template<size_t N, size_t... Is>
double distance_helper(const Point<N>& p1, const Point<N>& p2, std::index_sequence<Is...>) {
    int64_t sum = 0;
    // Calculate the sum of squared differences for each dimension
    ((sum += (std::get<Is>(p1) - std::get<Is>(p2)) * (std::get<Is>(p1) - std::get<Is>(p2))), ...);
    return std::sqrt(static_cast<double>(sum));
}

template<size_t N>
double distance(const Point<N>& p1, const Point<N>& p2) {
    return distance_helper(p1, p2, std::make_index_sequence<N>());
}


template<size_t N, size_t... Is>
int64_t in_distance_helper(const Point<N>& p1, const Point<N>& p2, std::index_sequence<Is...>) {
    int64_t sum = 0;
    // Calculate the sum of squared differences for each dimension
    ((sum += (std::get<Is>(p1) - std::get<Is>(p2)) * (std::get<Is>(p1) - std::get<Is>(p2))), ...);
    return sum;
}


template<size_t N>
bool in_distance(const Point<N>& p1, const Point<N>& p2, const int64_t radius_squared){
	return in_distance_helper<N>(p1, p2, std::make_index_sequence<N>()) <= radius_squared;	
}

template<size_t N>
bool in_distance(const Point<N>& p1, const Point<N>& p2, const int64_t radius_low_squared, const int64_t radius_high_squared){
	int64_t distance = in_distance_helper<N>(p1, p2, std::make_index_sequence<N>());
	return distance >= radius_low_squared && distance <= radius_high_squared;
}


template<size_t N, size_t... Is>
bool is_greater_helper(const int64_t(*arr)[N], const Point<N>& p, std::index_sequence<Is...>) {
    return ((std::get<Is>(p) > arr[Is][0]) && ...);
}


template<size_t N>
bool is_greater(const int64_t** arr, const Point<N>& constraint){
	return is_greater_helper(arr, constraint, std::make_index_sequence<N>());
}

template<size_t N, size_t... Is>
bool is_lower_helper(const int64_t(*arr)[N], const Point<N>& p, std::index_sequence<Is...>) {
    return ((std::get<Is>(p) < arr[Is][0]) && ...);
}


template<size_t N>
bool is_lower(const int64_t** arr, const Point<N>& constraint){
	return is_lower_helper(arr, constraint, std::make_index_sequence<N>());
}

template<size_t N, size_t... Is>
bool is_lower_equal_helper(const int64_t(*arr)[N], const Point<N>& p, std::index_sequence<Is...>) {
    return ((std::get<Is>(p) <= arr[Is][0]) && ...);
}


template<size_t N>
bool is_lower_equal(const int64_t** arr, const Point<N>& constraint){
	return is_lower_equal_helper(arr, constraint, std::make_index_sequence<N>());
}


template<size_t N, size_t... Is>
Point<N> constructPointFromIntArray(const std::array<const int64_t*, N>& arr, std::index_sequence<Is...>) {
    return Point<N>((arr[Is][0])...);
}

// Function to construct a Point<N> tuple from a const (int64_t*)[N] array
template<size_t N>
Point<N> constructPointFromIntArray(const std::array<const int64_t*, N>& arr) {
    return constructPointFromIntArray(arr, std::make_index_sequence<N>());
}

template<size_t N, size_t... Is>
void add_helper(Point<N>& p, int64_t a, std::index_sequence<Is...>){
	((std::get<Is>(p) += a),...);
}

template<size_t N>
Point<N> add(const Point<N>& p, int64_t a){
	Point<N> cpy = p;
	add_helper(cpy, a, std::make_index_sequence<N>());
	return std::move(cpy);
}


//assumes N tensors, and all dtypes are int64_t
template<size_t N>
std::vector<Point<N>> radius_points(const Tensor& points, const Point<N> point, const int64_t& radius_sq, const Point<N>& lower, const Point<N>& upper){
	std::vector<Point<N>> pts;
	const Tensor* beginT = reinterpret_cast<const Tensor*>(points.data_ptr());
	const Tensor* endT = beginT + points.numel();
	/* const int64_t* begin = reinterpret_cast<const int64_t*>(beginT->data_ptr()); */
	/* const int64_t* end = begin + beginT->numel(); */
	std::array<const int64_t*, N> array;
	std::array<const int64_t*, N> arrayEnd;
	for(uint32_t i = 0; i < N; ++i, ++beginT){
		array[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr());
		arrayEnd[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr_end());
	}
	for(;array[0] != arrayEnd[0];){
		if(is_lower_equal(array, lower)){break;}
		for(uint32_t i = 0; i < N; ++i)
			++array[i];
	}
	for(;array[0] != arrayEnd[0];){
		if(is_lower(array, upper)){break;}
		Point<N> current_point = constructPointFromIntArray(array);
		if(in_distance(point, current_point, radius_sq)){
			pts.emplace_back(std::move(current_point));
		}
		for(uint32_t i = 0; i < N; ++i){++array[i];}
	}
	return std::move(pts);

	
}


template<size_t N>
std::vector<Point<N>>& radius_points(const Tensor& points, const Point<N> point, const int64_t& radius_low_sq, const int64_t& radius_high_sq, const Point<N>& lower, const Point<N>& upper, std::vector<Point<N>>& pts){
	const Tensor* beginT = reinterpret_cast<const Tensor*>(points.data_ptr());
	const Tensor* endT = beginT + points.numel();
	/* const int64_t* begin = reinterpret_cast<const int64_t*>(beginT->data_ptr()); */
	/* const int64_t* end = begin + beginT->numel(); */
	std::array<const int64_t*, N> array;
	std::array<const int64_t*, N> arrayEnd;
	for(uint32_t i = 0; i < N; ++i, ++beginT){
		array[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr());
		arrayEnd[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr_end());
	}
	for(;array[0] != arrayEnd[0];){
		if(is_lower_equal(array, lower)){break;}
		for(uint32_t i = 0; i < N; ++i)
			++array[i];
	}
	for(;array[0] != arrayEnd[0];){
		if(is_lower(array, upper)){break;}
		Point<N> current_point = constructPointFromIntArray(array);
		if(in_distance(point, current_point, radius_low_sq, radius_high_sq)){
			pts.emplace_back(std::move(current_point));
		}
	}
	return pts;
	
}


// Function to generate points within a radius of 1 for a given dimension of the point
template<size_t N, size_t Index>
void generatePointsWithinRadiusOne(const Point<N>& center, std::vector<Point<N>>& result) {
    auto generateForDimension = [&](int64_t value, auto dimension) {
	if(value < 0){return;}
        constexpr size_t dim = decltype(dimension)::value;
        Point<N> p = center;
        std::get<dim>(p) = value;
        result.push_back(p);
    };

    // Generate points within a radius of 1 for each dimension
    ((generateForDimension(std::get<Index>(center) + 1, std::integral_constant<size_t, Index>{}),
      generateForDimension(std::get<Index>(center) - 1, std::integral_constant<size_t, Index>{})));}


// Recursive function to generate points within a radius of 1 for all dimensions of the point
template<size_t N, size_t... Is>
void generatePointsWithinRadiusRecursiveOne(const Point<N>& center, std::vector<Point<N>>& result, std::index_sequence<Is...>) {
    // Call generatePointsWithinRadius for each dimension
    (generatePointsWithinRadiusOne<N, Is>(center, result), ...);
}


template<size_t N>
std::vector<Point<N>> pointsWithinRadiusOne(const Point<N>& center) {
    std::vector<Point<N>> result;
    result.reserve(std::pow(N, N+1));

    // Generate points within a radius of 1 for all dimensions
    generatePointsWithinRadiusRecursiveOne(center, result, std::make_index_sequence<N>());

    return std::move(result);
}


//this is what chat gpt came up with, it is wrong, and only works for 2 dimensions
//(this was made to be correct by me)
//I am going to make a version that works for all dimensions
/* template<size_t N> */
/* unordered_point_set<N> generate_points_within_radius(const Point<N>& center, double lower_radius, double upper_radius) { */
/*     unordered_point_set<N> points; */
    
/*     // Loop through each dimension of the point */
/*     for (double i = -upper_radius; i <= upper_radius; ++i) { */
/*         for (double j = -upper_radius; j <= upper_radius; ++j) { */
/*             // Calculate the distance from the center */
/*             double distance = 0.0; */
/* 	    distance += std::pow(std::get<0>(center) + i - std::get<0>(center), 2); */
/* 	    distance += std::pow(std::get<1>(center) + j - std::get<1>(center), 2); */
/*             for (size_t k = 0; k < N; ++k) { */
/*                 distance += pow((std::get<k>(center) + i - std::get<k>(center)), 2); */
/*             } */
/*             distance = sqrt(distance); */
            
/*             // Check if the distance falls within the radius range */
/*             if (distance >= lower_radius && distance <= upper_radius) { */
/*                 // Create a new point and add it to the set */
/*                 Point<N> new_point = center; */
/* 		std::get<0>(new_point) += i; */
/* 		std::get<1>(new_point) += j; */
/*                 points.insert(new_point); */
/*             } */
/*         } */
/*     } */
/* } */


template<size_t N>
bool done(const std::array<int64_t,N>& indexes, const int64_t& upper_radius){
	if(indexes[0] > upper_radius)
		return true;
	return false;
	for(uint32_t i = 0; i < N; ++i){
		if(indexes[i] < upper_radius)
			return false;
	}
	return true;
}




template<size_t N>
void increment_indexes(std::array<int64_t,N>& indexes, const int64_t& upper_radius){
    // Start from the rightmost index
    size_t i = N - 1;
    // Increment the index at position i
    indexes[i]++;
    // If the index exceeds its limit, reset it to zero and carry over to the next index
    while (i > 0 && indexes[i] >= upper_radius) {
        indexes[i] = -upper_radius;
        indexes[--i]++;
    }
}

template<size_t N>
void increment_indexes(std::array<int64_t,N>& indexes, const int64_t& upper_radius, const int64_t& lower_radius){
    if(lower_radius == 0){increment_indexes(indexes, upper_radius);return;}
    // Start from the rightmost index
    size_t i = N - 1;
    // Increment the index at position i
    if(indexes[i] == -lower_radius){indexes[i] = lower_radius;return;}
    indexes[i]++;
    // If the index exceeds its limit, reset it to zero and carry over to the next index
    while (i > 0 && indexes[i] >= upper_radius) {
        indexes[i] = -upper_radius;
        indexes[--i]++;
	if(indexes[i] == (-lower_radius)+1){indexes[i] = lower_radius;return;}
    }
}


template<size_t N, size_t... Is>
void add_helper(Point<N>& p, const std::array<int64_t, N>& a, std::index_sequence<Is...>){
	((std::get<Is>(p) += a[Is]), ...);
}

template<size_t N>
Point<N> add(const Point<N>& p, const std::array<int64_t, N>& a){
	Point<N> cpy = p;
	add_helper(cpy, a, std::make_index_sequence<N>());
	return std::move(cpy);
}


template<size_t N>
unordered_point_set<N> Points<N>::generate_all_points_within_radius(const Point<N>& center, int64_t lower_radius, int64_t upper_radius) const{
	unordered_point_set<N> points;

	std::array<int64_t, N> indexes;
	for(uint32_t i = 0; i < N; ++i){
		indexes[i] = -upper_radius;
	}

	int64_t lower_sq = lower_radius * lower_radius;
	int64_t higher_sq = upper_radius * upper_radius;
	uint32_t i = 0;
	while(!done(indexes, upper_radius)){
		Point<N> current = add(center, indexes);
		if(in_distance<N>(center, current, lower_sq, higher_sq)){ //makes sure the 2 points are within the radius
			points.insert(current);
		}
		increment_indexes(indexes, upper_radius, lower_radius);
	}
	return std::move(points);

}




template<size_t N>
std::vector<Point<N> > Points<N>::generatePoints() const{
	utils::throw_exception(coords.dtype == DType::TensorObj, "Expected coordinates to have a dtype of tensor but got $", coords.dtype);
	utils::throw_exception(coords.numel() == N, "Expected to have $ tensors in coords but got $", N, coords.numel());
	const Tensor* ts = reinterpret_cast<const Tensor*>(coords.data_ptr());
	std::vector<Point<N> > points;
	points.reserve(ts[0].numel());
	std::array<const int64_t*, N> array;
	std::array<const int64_t*, N> arrayEnd;
	for(uint32_t i = 0; i < N; ++i, ++ts){
		array[i] = reinterpret_cast<const int64_t*>(ts->data_ptr());
		arrayEnd[i] = reinterpret_cast<const int64_t*>(ts->data_ptr_end());
	}

	while(array[0] != arrayEnd[0]){
		points.push_back(constructPointFromIntArray(array));
		for(uint32_t i = 0; i < N; ++i){++array[i];}
	}
	return std::move(points);

}

template class Points<1>;
template class Points<2>;
template class Points<3>;
template class Points<4>;
template class Points<5>;
template class Points<6>;
template class Points<7>;
template class Points<8>;
template class Points<9>;
template class Points<10>;
template class Points<11>;
template class Points<12>;
template class Points<13>;
template class Points<14>;
template class Points<15>;
template class Points<16>;
template class Points<17>;
template class Points<18>;
template class Points<19>;
template class Points<20>;
template class Points<21>;
template class Points<22>;
template class Points<23>;
template class Points<24>;
template class Points<25>;
template class Points<26>;
template class Points<27>;
template class Points<28>;
template class Points<29>;
template class Points<30>;

}
}
