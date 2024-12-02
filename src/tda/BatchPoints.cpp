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
#include "BatchPoints.h"


namespace nt{
namespace tda{

BatchPoints::BatchPoints(const Tensor& t, const uint8_t point)
	:dim(0), point(point), last(-1)
{
	if(t.dtype == DType::uint8){
		original = t.clone();
	}else{
		original = t.to_dtype(DType::uint8);
	}
	/* coords = functional::where(original == point); */
	const_cast<int64_t&>(dim) = t.dims()-1;
}


BatchPoints::BatchPoints(Tensor&& t, const uint8_t point)
	:original(std::move(t)), dim(0), point(point), last(-1)
{
	const_cast<int64_t&>(dim) = t.dims()-1;
	
}

BatchPoints::BatchPoints(const Tensor& t, const uint8_t point, const int64_t dim)
	:dim(dim), point(point), last(-1)
{
	if(t.dtype == DType::uint8){
		original = t.clone();
	}else{
		original = t.to_dtype(DType::uint8);
	}
	utils::throw_exception(original.dims() == (dim+1), "Expected to have the input tensor be one greater than the input dim ($) but got $", dim, original.dims());
	/* coords = std::move(functional::where(original == point)); */	
}

BatchPoints::BatchPoints(Tensor&& t, const uint8_t point, const int64_t dim)
	:original(std::move(t)), dim(dim), point(point), last(-1)
{
	utils::throw_exception(original.dims() == (dim+1), "Expected to have the input tensor be one greater than the input dim ($) but got $", dim, original.dims());


}


//expected by this point there have been enough checks that p1 and p2 are of the same size
inline int64_t distance_n_sqrt(const Point& p1, const Point& p2) noexcept {
    int64_t sum = 0;
    // Calculate the sum of squared differences for each dimension
    for(size_t i = 0; i < p1.size(); ++i){
	sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sum;
}

inline double distance(const Point& p1, const Point& p2) {
    int64_t sum = 0;
    utils::THROW_EXCEPTION(p1.size() == p2.size(), "Expected points to have the same size for distance");
    // Calculate the sum of squared differences for each dimension
    for(size_t i = 0; i < p1.size(); ++i){
	sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return std::sqrt(static_cast<double>(sum));
}



inline bool in_distance(const Point& p1, const Point& p2, const double radius_squared){
	utils::THROW_EXCEPTION(p1.size() == p2.size(), "Expected points to have the same size for in distance");
	return distance_n_sqrt(p1, p2) <= radius_squared;
}

inline bool in_distance(const Point& p1, const Point& p2, const double radius_low_squared, const double radius_high_squared){
	utils::THROW_EXCEPTION(p1.size() == p2.size(), "Expected points to have the same size for in distance");
	int64_t distance = distance_n_sqrt(p1, p2);
	return distance >= radius_low_squared && distance <= radius_high_squared;
}


inline bool is_greater(const int64_t** arr, const Point& constraint){
	for(int64_t i = 0; i < constraint.size(); ++i){
		if(constraint[i] <= arr[i][0]){return false;}
	}
	return true;
}


inline bool is_lower(const int64_t** arr, const Point& constraint){
	for(int64_t i = 0; i < constraint.size(); ++i){
		if(constraint[i] >= arr[i][0]){return false;}
	}
	return true;
}




inline bool is_lower_equal(const int64_t** arr, const Point& constraint){
	for(int64_t i = 0; i < constraint.size(); ++i){
		if(constraint[i] > arr[i][0]){return false;}
	}
	return true;
}



inline Point constructPointFromIntArray(const std::vector<const int64_t*>& arr){
	Point out(arr.size());
	for(int64_t i = 0; i < arr.size(); ++i)
		out[i] = arr[i][0];
	return std::move(out);
}





//assumes all dtypes are int64_t
inline std::vector<Point> radius_points(const Tensor& points, const Point point, const double& radius_sq, const Point& lower, const Point& upper){
	const int64_t N = points.dims();
	utils::THROW_EXCEPTION(point.size() == N, "Expected to get radius points in the same dimension as the incoming tensor but got $ and $", points.dims(), point.size());
	std::vector<Point> pts;
	const Tensor* beginT = reinterpret_cast<const Tensor*>(points.data_ptr());
	const Tensor* endT = beginT + points.numel();
	/* const int64_t* begin = reinterpret_cast<const int64_t*>(beginT->data_ptr()); */
	/* const int64_t* end = begin + beginT->numel(); */
	std::vector<const int64_t*> array(N);
	std::vector<const int64_t*> arrayEnd(N);
	for(uint32_t i = 0; i < N; ++i, ++beginT){
		array[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr());
		arrayEnd[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr_end());
	}
	for(;array[0] != arrayEnd[0];){
		if(is_lower_equal(array.data(), lower)){break;}
		//the above checks if the variable lower is lower than the point
		//if not, then the point currently in the array data is lower
		//and move onto the next point
		for(int64_t i = 0; i < N; ++i)
			++array[i];
	}
	for(;array[0] != arrayEnd[0];){

		if(is_lower(array.data(), upper)){break;}
		//if upper is lowerr than the array data, then we have gone outside the scope and break
		Point current_point = constructPointFromIntArray(array);
		//if it is within the correct radius add it
		if(in_distance(point, current_point, radius_sq)){
			pts.emplace_back(std::move(current_point));
		}
		//go to the next point
		for(int64_t i = 0; i < N; ++i){++array[i];}
	}
	return std::move(pts);

	
}


inline std::vector<Point>& radius_points(const Tensor& points, const Point point, const double& radius_low_sq, const double& radius_high_sq, const Point& lower, const Point& upper, std::vector<Point>& pts){
	const int64_t N = points.dims();
	utils::THROW_EXCEPTION(point.size() == N, "Expected to get radius points in the same dimension as the incoming tensor but got $ and $", points.dims(), point.size());
	utils::THROW_EXCEPTION(N == lower.size() && N == upper.size(), "Expected lower and upper to have dimension $ but got $ and $", N, lower.size(), upper.size());
	const Tensor* beginT = reinterpret_cast<const Tensor*>(points.data_ptr());
	const Tensor* endT = beginT + points.numel();
	/* const int64_t* begin = reinterpret_cast<const int64_t*>(beginT->data_ptr()); */
	/* const int64_t* end = begin + beginT->numel(); */
	std::vector<const int64_t*> array(N);
	std::vector<const int64_t*> arrayEnd(N);
	for(uint32_t i = 0; i < N; ++i, ++beginT){
		array[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr());
		arrayEnd[i] = reinterpret_cast<const int64_t*>(beginT->data_ptr_end());
	}
	for(;array[0] != arrayEnd[0];){
		if(is_lower_equal(array.data(), lower)){break;}
		//the above checks if the variable lower is lower than the point
		//if not, then the point currently in the array data is lower
		//and move onto the next point
		for(int64_t i = 0; i < N; ++i)
			++array[i];
	}
	for(;array[0] != arrayEnd[0];){
		if(is_lower(array.data(), upper)){break;}
		Point current_point = constructPointFromIntArray(array);
		if(in_distance(point, current_point, radius_low_sq, radius_high_sq)){
			pts.emplace_back(std::move(current_point));
		}
	}
	return pts;
	
}


// Function to generate points within a radius of 1 for a given dimension of the point
inline void generatePointsWithinRadiusOne(const Point& center, std::vector<Point>& result, const int64_t& index) {
    int64_t value_a = center[index] + 1;
    int64_t value_b = center[index] - 1;
    if(value_a >= 0){
	Point p = center;
	p[index] = value_a;
	result.push_back(p);
    }
    if(value_b >= 0){
	Point p = center;
	p[index] = value_b;
	result.push_back(p);
    }
}

// Recursive function to generate points within a radius of 1 for all dimensions of the point
inline void generatePointsWithinRadiusRecursiveOne(const Point& center, std::vector<Point>& result) {
    for(int64_t i = 0; i < center.size(); ++i){
	generatePointsWithinRadiusOne(center, result, i);
    }
}


inline std::vector<Point> pointsWithinRadiusOne(const Point& center) {
    std::vector<Point> result;
    result.reserve(std::pow(center.size(), center.size()+1));

    // Generate points within a radius of 1 for all dimensions
    generatePointsWithinRadiusRecursiveOne(center, result);

    return std::move(result);
}




inline bool done(const std::vector<int64_t>& indexes, const double& upper_radius){
	if(indexes[0] > upper_radius)
		return true;
	return false;
	const size_t N = indexes.size();
	for(uint32_t i = 0; i < N; ++i){
		if(indexes[i] < upper_radius)
			return false;
	}
	return true;
}




inline void increment_indexes(std::vector<int64_t>& indexes, const double& upper_radius){
    const size_t N = indexes.size();
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

inline void increment_indexes(std::vector<int64_t>& indexes, const double& upper_radius, const double& lower_radius){
    if(lower_radius == 0){increment_indexes(indexes, upper_radius);return;}
    const size_t N = indexes.size();
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

inline Point add(const Point& p, const std::vector<int64_t>& a){
	utils::THROW_EXCEPTION(p.size() == a.size(), "Cannot add vector to point of different dimensions $ and $", p.size(), a.size());
	Point cpy = p;
	for(int64_t i = 0; i < a.size(); ++i)
		cpy[i] += a[i];
	return std::move(cpy);
}


void BatchPoints::generate_all_points_within_radius(const Point& center, double lower_radius, double upper_radius, const int64_t index, unordered_point_set& points) const{
	/* unordered_point_set points; */
	
	const size_t N = center.size();
	std::vector<int64_t> indexes(N);
	for(uint32_t i = 0; i < N; ++i){
		indexes[i] = -upper_radius;
	}

	double lower_sq = lower_radius * lower_radius;
	double higher_sq = upper_radius * upper_radius;
	uint32_t i = 0;
	while(!done(indexes, upper_radius)){
		Point current = add(center, indexes);
		if(in_distance(center, current, lower_sq, higher_sq)){ //makes sure the 2 points are within the radius
			points.insert(current);
		}
		increment_indexes(indexes, upper_radius, lower_radius);
	}
}


//used to get the first and last index per batch
void firstLastIndex(const int64_t* data, const int64_t* data_end, int64_t& begin, int64_t& end, int64_t element){
	bool first_found = false;
	int64_t index = 0;
	for(;data != data_end; ++data, ++index){
		if(*data == element){
			if(!first_found){begin = index;}
			end = index;
		}
	}
}

void commence_generation(const Tensor& original, const uint8_t& point, std::vector<std::vector<Point>>& pts){
	const Tensor split = original.split_axis(0);
	const Tensor* t_begin = reinterpret_cast<const Tensor*>(split.data_ptr());
	const Tensor* t_end = t_begin + split.numel();
	pts.reserve(split.numel());
	uint32_t counter = 0;
	const int64_t& batches = split.numel();
	for(;t_begin != t_end; ++t_begin, ++counter){
		utils::printProgressBar(counter, batches);
		Tensor coords = functional::where(*t_begin == point);
		const int64_t& N = coords.numel();
		const Tensor* ts = reinterpret_cast<const Tensor*>(coords.data_ptr());
		std::vector<Point> points;
		points.reserve(ts[0].numel());
		std::vector<const int64_t*> array(N);
		std::vector<const int64_t*> arrayEnd(N);
		for(int i = 0; i < N; ++i, ++ts){
			array[i] = reinterpret_cast<const int64_t*>(ts->data_ptr());
			arrayEnd[i] = reinterpret_cast<const int64_t*>(ts->data_ptr_end());
		}
		while(array[0] != arrayEnd[0]){
			points.push_back(constructPointFromIntArray(array));
			for(uint32_t i = 0; i < N; ++i){++array[i];}
		}
		pts.push_back(std::move(points));
	}
}


void commence_single_generation(const Tensor& original, const uint8_t& point, std::vector<Point>& points, const int64_t& batch){
	points.clear();
	const Tensor& current = original[batch];
	Tensor coords = functional::where(current == point);
	const int64_t& N = coords.numel();
	const Tensor* ts = reinterpret_cast<const Tensor*>(coords.data_ptr());
	points.reserve(ts[0].numel());
	std::vector<const int64_t*> array(N);
	std::vector<const int64_t*> arrayEnd(N);
	for(int i = 0; i < N; ++i, ++ts){
		array[i] = reinterpret_cast<const int64_t*>(ts->data_ptr());
		arrayEnd[i] = reinterpret_cast<const int64_t*>(ts->data_ptr_end());
	}
	while(array[0] != arrayEnd[0]){
		points.push_back(constructPointFromIntArray(array));
		for(uint32_t i = 0; i < N; ++i){++array[i];}
	}
}

//the generatePoints function for each batch index
const std::vector<Point>& BatchPoints::generatePoints(const int64_t batch) const{
	if(last == batch){return pts;}
	commence_single_generation(this->original, this->point, const_cast<std::vector<Point>&>(pts), batch);
	/* if(!this->pts.empty()){return pts[dim];} */
	//otherwize generate the points
	/* commence_generation(this->original, this->point, const_cast<std::vector<std::vector<Point>>&>(pts)); */
	return pts;

}

const int64_t BatchPoints::count_points() const noexcept{
	return std::count(reinterpret_cast<const uint8_t*>(original.data_ptr()),
			reinterpret_cast<const uint8_t*>(original.data_ptr_end()),
			this->point);
}

}}
