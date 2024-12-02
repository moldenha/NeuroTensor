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

/* void points_2d::generatePoints(const uint8_t& point){ */
/* 	utils::throw_exception(original.dims() == 2, "expected Tensor dims to be 2 but got $ with shape $", original.dims(), original.shape()); */
/* 	Tensor w = functional::where(points == point); */
/* 	Tensor x = w[0].item<Tensor>(); */
/* 	Tensor y = w[1].item<Tensor>(); */
/* 	utils::throw_exception(x.dtype == DType::int64 && y.dtype == DType::int64, "Expected to make Tensors of dtype int64 but got $ and $ instead", x.dtype, y.dtype); */
/* 	this->points.reserve(x.numel()); */
/* 	int64_t* x_begin = reinterpret_cast<int64_t*>(x.data_ptr()); */
/* 	int64_t* y_begin = reinterpret_cast<int64_t*>(y.data_ptr()); */
/* 	int64_t* end = x_begin + x.numel(); */
/* 	for(;x_begin != end; ++x_begin, ++y_begin){ */
/* 		this->points.push_back(point_2d(*x, *y)); */
/* 	} */
/* } */

points_2d::points_2d(const nt::Tensor& t, const uint8_t point) // point refers to the number in the tensor that it is looking for
	:original(t.to_dtype(DType::uint8)), coords(functional::where(t == point))

{
	utils::throw_exception(t.dims() == 2, "\nRuntimeError: Expected tensor to have 2 dims but got $", t.dims());

}

points_2d::points_2d(nt::Tensor&& t, const uint8_t point) // point refers to the number in the tensor that it is looking for
	:original(std::move(t))

{
	utils::throw_exception(original.dims() == 2, "\nRuntimeError: Expected tensor to have 2 dims but got $", original.dims());
	coords = std::move(functional::where(original == point));
	std::cout<<"made coords"<<std::endl;

}

std::vector<Point2d> radius_points(const Tensor& x, const Tensor& y, const Point2d& point, const int64_t& radius, const int64_t& lower_x, const int64_t& upper_x, const int64_t& lower_y, const int64_t& upper_y){

	return x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>(
			[&point, &radius, &lower_x, &upper_x, &lower_y, &upper_y](auto a_begin, auto a_end, auto b_begin) -> std::vector<Point2d>{
			std::vector<Point2d> points;
			points.reserve(20);
			const int64_t a = point.first;
			const int64_t b = point.second;
			for(;a_begin != a_end; ++a_begin, ++b_begin){
				if(*a_begin >= lower_x && *b_begin >= lower_y)
					break;
			}
			for(;a_begin != a_end; ++a_begin, ++b_begin){
				if(*b_begin > upper_y || *a_begin > upper_x)
					break;
				const int64_t xp = *a_begin - a;
				const int64_t yp = *b_begin - b;
				const int64_t rhs = (xp*xp) + (yp*yp);
				if(rhs <= radius){
					points.emplace_back(Point2d(*a_begin, *b_begin));
				}
			}
			return std::move(points);
			}, y.arr_void());
}

void radius_points_b(const Tensor& x, const Tensor& y, const Point2d& point, const int64_t& radius_high, const int64_t& radius_low, const int64_t& lower_x, const int64_t& upper_x, const int64_t& lower_y, const int64_t& upper_y, std::vector<Point2d >& pairs){

	 x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>(
			[&point, &radius_high,&radius_low, &lower_x, &upper_x, &lower_y, &upper_y, &pairs](auto a_begin, auto a_end, auto b_begin){
			/* bool include_next_to = (radius_low <= 0 */ 
			const int64_t a = point.first;
			const int64_t b = point.second;
			for(;a_begin != a_end; ++a_begin, ++b_begin){
				if(*a_begin >= lower_x && *b_begin >= lower_y)
					break;
			}
			for(;a_begin != a_end; ++a_begin, ++b_begin){
				if(*a_begin > upper_x && *b_begin > upper_y)
					break;
				/* if(*b_begin > upper_y){ */
				/* 	const auto val = *a_begin; */
				/* 	while(*a_begin == val && a_begin != a_end){++a_begin;++b_begin;} */
				/* 	if(a_begin == a_end){break;} */
				/* } */
				const int64_t xp = *a_begin - a;
				const int64_t yp = *b_begin - b;
				const int64_t rhs = (xp*xp) + (yp*yp);
				if(rhs >= radius_low && rhs <= radius_high){
					pairs.emplace_back(Point2d(*a_begin, *b_begin));
				}
			}
			}, y.arr_void());
}


inline bool validRadiusCheck(const int64_t& X1, const int64_t& Y1, const int64_t& X2, const int64_t& Y2, const int64_t& radius_high_sq, const int64_t& radius_low_sq){
	const int64_t x = (X1 - X2);
	const int64_t y = (Y1 - Y2);
	const int64_t rhs = (x * x) + (y * y);
	return rhs >= radius_low_sq && rhs <= radius_high_sq;
}

inline int64_t validRadiusRHS(const int64_t& X1, const int64_t& Y1, const int64_t& X2, const int64_t& Y2){
	const int64_t x = (X1 - X2);
	const int64_t y = (Y1 - Y2);
	return (x * x) + (y * y);
}


//this function assumes the radius has already been squared coming in
//currently able to get the max y per x
void radius_points_filtered(const Tensor& x, const Tensor& y, const Point2d& point, const int64_t& radius_high, const int64_t& radius_low, const int64_t& lower_x, const int64_t& upper_x, const int64_t& lower_y, const int64_t& upper_y, std::vector<Point2d >& pairs){
	//first I am going to grab all the max and min Y's per x:
	std::map<int64_t, std::vector<int64_t> > yMap; //it goes yMap[y] gets a list of corresponding x's and then it gets the max x
	std::map<int64_t, std::vector<int64_t> > xMap; //it goes yMap[y] gets a list of corresponding x's and then it gets the max x
	
	x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>(
			[&point, &radius_high,&radius_low, &lower_x, &upper_x, &lower_y, &upper_y, &yMap, &xMap](auto a_begin, auto a_end, auto b_begin){
			const int64_t a = point.first;
			const int64_t b = point.second;
			for(;a_begin != a_end; ++a_begin, ++b_begin){
				if(*a_begin >= lower_x && *b_begin >= lower_y)
					break;
			}

			for(;a_begin != a_end; ++a_begin, ++b_begin){
				if(*a_begin > upper_x && *b_begin > upper_y)
					break;
				if(*b_begin < lower_y)
					continue;
				if(validRadiusCheck(*a_begin, *b_begin, a, b, radius_high, radius_low)){
					xMap[*a_begin].push_back(*b_begin);
					yMap[*b_begin].push_back(*a_begin);
				}
				/* if(validRadiusCheck(*a_begin, a, *b_begin, b, radius_high, radius_low)){ */
				/* 	xMap[*a_begin].push_back(*b_begin); */
				/* 	yMap[*b_begin].push_back(*a_begin); */
				/* } */
			}
			}, y.arr_void());
	/* return; */
	unordered_point2d_set orderedPoints;
	for(const auto& [Y, Xs] : yMap){
		const int64_t minX = *std::min_element(Xs.cbegin(), Xs.cend());
		const int64_t maxX = *std::max_element(Xs.cbegin(), Xs.cend());
		orderedPoints.insert(Point2d(minX, Y));
		orderedPoints.insert(Point2d(maxX, Y));
	}
	//now I am going to grab all the max and min X's per y:
	for(const auto& [X, Ys] : xMap){
		const uint64_t& minY = *std::min_element(Ys.cbegin(), Ys.cend());
		const uint64_t& maxY = *std::max_element(Ys.cbegin(), Ys.cend());
		orderedPoints.insert(Point2d(X, minY));
		orderedPoints.insert(Point2d(X, maxY));
	}
	pairs.reserve(orderedPoints.size());
	pairs.insert(pairs.end(), orderedPoints.begin(), orderedPoints.end());
}


Tensor to_tensor(const std::vector<Point2d >& points){
	Tensor xy({2}, DType::TensorObj);
	Tensor* xy_p = reinterpret_cast<Tensor*>(xy.data_ptr());
	*xy_p = Tensor({static_cast<unsigned int>(points.size())}, DType::int64);
	*(xy_p + 1) = Tensor({static_cast<unsigned int>(points.size())}, DType::int64);
	
	Tensor& x = *xy_p;
	Tensor& y = *(xy_p + 1);
	int64_t* x_begin = reinterpret_cast<int64_t*>(x.data_ptr());
	int64_t* x_end = x_begin + x.numel();
	int64_t* y_begin = reinterpret_cast<int64_t*>(y.data_ptr());
	auto p_begin = points.cbegin();
	for(;x_begin != x_end; ++x_begin, ++y_begin, ++p_begin){
		*x_begin = p_begin->first;
		*y_begin = p_begin->second;
	}
	return xy;
}

Tensor points_2d::in_radius(Point2d point, int64_t radius) const{
	//distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
	//I am given x2 and y2
	//a = point.first;
	//b = point.secod;
	const int64_t lower_x = std::max(point.first - radius, (int64_t)0);
	const int64_t upper_x = std::min(point.first + radius, static_cast<int64_t>(original.shape()[-2]));
	const int64_t lower_y = std::max(point.second - radius, (int64_t)0);
	const int64_t upper_y = std::min(point.second + radius, static_cast<int64_t>(original.shape().back()));
	const Tensor& x = X_points();
	const Tensor& y = Y_points();
	
	Tensor w = functional::where((x >= lower_x && x <= upper_x) && (y >= lower_y && y <= upper_y));
	Tensor nx = x[w];
	Tensor ny = y[w];
	radius *= radius;
	std::vector<Point2d> radi = radius_points(nx, ny, point, radius, lower_x, upper_x, lower_y, upper_y);

	return to_tensor(radi);
}

std::vector<Point2d> points_2d::in_radius_vec(const Point2d& point, int64_t radius) const{
	//distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
	//I am given x2 and y2
	//a = point.first;
	//b = point.secod;
	
	const int64_t lower_x = std::max(point.first - radius, (int64_t)0);
	const int64_t upper_x = std::min(point.first + radius, static_cast<int64_t>(original.shape()[-2]));
	const int64_t lower_y = std::max(point.second - radius, (int64_t)0);
	const int64_t upper_y = std::min(point.second + radius, static_cast<int64_t>(original.shape().back()));
	const Tensor& x = X_points();
	const Tensor& y = Y_points();
	
	/* Tensor w = functional::where((x >= lower_x && x <= upper_x) && (y >= lower_y && y <= upper_y)); */
	/* Tensor nx = x[w]; */
	/* Tensor ny = y[w]; */
	radius *= radius;
	return radius_points(x, y, point, radius, lower_x, upper_x, lower_y, upper_y);
}

unordered_point2d_set points_2d::in_radius_one(const Point2d& point) const{
	const Tensor& x = X_points();
	const Tensor& y = Y_points();
	unordered_point2d_set points;
	int64_t up_x_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
	int64_t up_x_y = point.second;
	int64_t up_y_x = point.first;
	int64_t up_y_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

	int64_t down_x_x = std::max(point.first - 1, static_cast<int64_t>(0));
	int64_t down_x_y = point.second;
	int64_t down_y_x = point.first;
	int64_t down_y_y = std::max(point.second - 1, static_cast<int64_t>(0));

	int64_t up_left_x = std::max(point.first - 1, static_cast<int64_t>(0));
	int64_t up_left_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

	int64_t up_right_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
	int64_t up_right_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

	int64_t down_left_x = std::max(point.first - 1, static_cast<int64_t>(0));
	int64_t down_left_y = std::max(point.second - 1, static_cast<int64_t>(0));

	int64_t down_right_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
	int64_t down_right_y = std::max(point.second - 1, static_cast<int64_t>(0));
	uint32_t y_end = y.numel();
	x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>(
			[&up_x_x, &up_x_y, &up_y_x, &up_y_y, &down_x_x, &down_x_y, &down_y_x, &down_y_y,
			&up_left_x, &up_left_y, &up_right_x, &up_right_y, &down_left_x, &down_left_y,
			&down_right_y, &down_right_x, &y_end, &points](auto a_begin, auto a_end, auto b_begin){
			/* bool include_next_to = (radius_low <= 0 */ 
			auto b_end = b_begin + y_end;
			if(std::find(a_begin, a_end, up_x_x) != a_end && std::find(b_begin, b_end, up_x_y) != b_end)
				points.insert(Point2d(up_x_x, up_x_y));
			if(std::find(a_begin, a_end, up_y_x) != a_end && std::find(b_begin, b_end, up_y_y) != b_end)
				points.insert(Point2d(up_y_x, up_y_y));
			
			if(std::find(a_begin, a_end, down_x_x) != a_end && std::find(b_begin, b_end, down_x_y) != b_end)
				points.insert(Point2d(down_x_x, down_x_y));
			if(std::find(a_begin, a_end, down_y_x) != a_end && std::find(b_begin, b_end, down_y_y) != b_end)
				points.insert(Point2d(down_y_x, down_y_y));

			if(std::find(a_begin, a_end, up_left_x) != a_end && std::find(b_begin, b_end, up_left_y) != b_end)
				points.insert(Point2d(up_left_x, up_left_y));
			if(std::find(a_begin, a_end, up_right_x) != a_end && std::find(b_begin, b_end, up_right_y) != b_end)
				points.insert(Point2d(up_right_x, up_right_y));

			if(std::find(a_begin, a_end, down_left_x) != a_end && std::find(b_begin, b_end, down_left_y) != b_end)
				points.insert(Point2d(down_left_x, down_left_y));
			if(std::find(a_begin, a_end, down_right_x) != a_end && std::find(b_begin, b_end, down_right_y) != b_end)
				points.insert(Point2d(down_right_x, down_right_y));
			}, y.arr_void());
	return std::move(points);




}

std::vector<Point2d> points_2d::in_radius_vec(const Point2d& point, int64_t radius_high, int64_t radius_low) const{
	//distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
	//I am given x2 and y2
	//a = point.first;
	//b = point.secod;
	const int64_t lower_x = std::max(point.first - radius_high, (int64_t)0);
	const int64_t upper_x = std::min(point.first + radius_high, static_cast<int64_t>(original.shape()[-2]));
	const int64_t lower_y = std::max(point.second - radius_high, (int64_t)0);
	const int64_t upper_y = std::min(point.second + radius_high, static_cast<int64_t>(original.shape().back()));
	/* const int64_t lower_x = 0; */
	/* const int64_t lower_y = 0; */
	/* const int64_t upper_x = static_cast<int64_t>(original.shape()[-2]); */
	/* const int64_t upper_y = static_cast<int64_t>(original.shape().back()); */
	const Tensor& x = X_points();
	const Tensor& y = Y_points();
	std::vector<Point2d> points;
	points.reserve(20);
	if(radius_low <= 1 && radius_high >= 1){
		int64_t up_x_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
		int64_t up_x_y = point.second;
		int64_t up_y_x = point.first;
		int64_t up_y_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

		int64_t down_x_x = std::max(point.first - 1, static_cast<int64_t>(0));
		int64_t down_x_y = point.second;
		int64_t down_y_x = point.first;
		int64_t down_y_y = std::max(point.second - 1, static_cast<int64_t>(0));

		int64_t up_left_x = std::max(point.first - 1, static_cast<int64_t>(0));
		int64_t up_left_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

		int64_t up_right_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
		int64_t up_right_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

		int64_t down_left_x = std::max(point.first - 1, static_cast<int64_t>(0));
		int64_t down_left_y = std::max(point.second - 1, static_cast<int64_t>(0));

		int64_t down_right_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
		int64_t down_right_y = std::max(point.second - 1, static_cast<int64_t>(0));
		uint32_t y_end = y.numel();
		x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>(
			[&up_x_x, &up_x_y, &up_y_x, &up_y_y, &down_x_x, &down_x_y, &down_y_x, &down_y_y,
			&up_left_x, &up_left_y, &up_right_x, &up_right_y, &down_left_x, &down_left_y,
			&down_right_y, &down_right_x, &y_end, &points](auto a_begin, auto a_end, auto b_begin){
			/* bool include_next_to = (radius_low <= 0 */ 
			auto b_end = b_begin + y_end;
			if(std::find(a_begin, a_end, up_x_x) != a_end && std::find(b_begin, b_end, up_x_y) != b_end)
				points.push_back(Point2d(up_x_x, up_x_y));
			if(std::find(a_begin, a_end, up_y_x) != a_end && std::find(b_begin, b_end, up_y_y) != b_end)
				points.push_back(Point2d(up_y_x, up_y_y));
			
			if(std::find(a_begin, a_end, down_x_x) != a_end && std::find(b_begin, b_end, down_x_y) != b_end)
				points.push_back(Point2d(down_x_x, down_x_y));
			if(std::find(a_begin, a_end, down_y_x) != a_end && std::find(b_begin, b_end, down_y_y) != b_end)
				points.push_back(Point2d(down_y_x, down_y_y));

			if(std::find(a_begin, a_end, up_left_x) != a_end && std::find(b_begin, b_end, up_left_y) != b_end)
				points.push_back(Point2d(up_left_x, up_left_y));
			if(std::find(a_begin, a_end, up_right_x) != a_end && std::find(b_begin, b_end, up_right_y) != b_end)
				points.push_back(Point2d(up_right_x, up_right_y));

			if(std::find(a_begin, a_end, down_left_x) != a_end && std::find(b_begin, b_end, down_left_y) != b_end)
				points.push_back(Point2d(down_left_x, down_left_y));
			if(std::find(a_begin, a_end, down_right_x) != a_end && std::find(b_begin, b_end, down_right_y) != b_end)
				points.push_back(Point2d(down_right_x, down_right_y));


			}, y.arr_void());


		int64_t current_x = point.first;
		int64_t current_y = point.second;
	}
	if(radius_high == 1){return std::move(points);}

	
	/* Tensor w = functional::where((x >= lower_x && x <= upper_x) && (y >= lower_y && y <= upper_y)); */
	/* Tensor nx = x[w]; */
	/* Tensor ny = y[w]; */
	radius_low *= radius_low;
	radius_high *= radius_high;
	radius_points_b(x, y, point, radius_high, radius_low, lower_x, upper_x, lower_y, upper_y, points);
	return std::move(points);
}

std::vector<Point2d> points_2d::in_radius_filtered(const Point2d& point, int64_t radius_high, int64_t radius_low) const{
	const int64_t lower_x = std::max(point.first - radius_high, (int64_t)0);
	const int64_t upper_x = std::min(point.first + radius_high, static_cast<int64_t>(original.shape()[-2]));
	const int64_t lower_y = std::max(point.second - radius_high, (int64_t)0);
	const int64_t upper_y = std::min(point.second + radius_high, static_cast<int64_t>(original.shape().back()));
	/* const int64_t lower_x = 0; */
	/* const int64_t lower_y = 0; */
	/* const int64_t upper_x = static_cast<int64_t>(original.shape()[-2]); */
	/* const int64_t upper_y = static_cast<int64_t>(original.shape().back()); */
	const Tensor& x = X_points();
	const Tensor& y = Y_points();
	std::vector<Point2d> points;
	points.reserve(20);
	if(radius_low <= 1 && radius_high >= 1){
		int64_t up_x_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
		int64_t up_x_y = point.second;
		int64_t up_y_x = point.first;
		int64_t up_y_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

		int64_t down_x_x = std::max(point.first - 1, static_cast<int64_t>(0));
		int64_t down_x_y = point.second;
		int64_t down_y_x = point.first;
		int64_t down_y_y = std::max(point.second - 1, static_cast<int64_t>(0));

		int64_t up_left_x = std::max(point.first - 1, static_cast<int64_t>(0));
		int64_t up_left_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

		int64_t up_right_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
		int64_t up_right_y = std::min(point.second + 1, static_cast<int64_t>(original.shape()[-1]));

		int64_t down_left_x = std::max(point.first - 1, static_cast<int64_t>(0));
		int64_t down_left_y = std::max(point.second - 1, static_cast<int64_t>(0));

		int64_t down_right_x = std::min(point.first + 1, static_cast<int64_t>(original.shape()[-2]));
		int64_t down_right_y = std::max(point.second - 1, static_cast<int64_t>(0));
		uint32_t y_end = y.numel();
		x.arr_void().cexecute_function<WRAP_DTYPES<DTypeEnum<DType::int64>>>(
			[&up_x_x, &up_x_y, &up_y_x, &up_y_y, &down_x_x, &down_x_y, &down_y_x, &down_y_y,
			&up_left_x, &up_left_y, &up_right_x, &up_right_y, &down_left_x, &down_left_y,
			&down_right_y, &down_right_x, &y_end, &points](auto a_begin, auto a_end, auto b_begin){
			/* bool include_next_to = (radius_low <= 0 */ 
			auto b_end = b_begin + y_end;
			if(std::find(a_begin, a_end, up_x_x) != a_end && std::find(b_begin, b_end, up_x_y) != b_end)
				points.push_back(Point2d(up_x_x, up_x_y));
			if(std::find(a_begin, a_end, up_y_x) != a_end && std::find(b_begin, b_end, up_y_y) != b_end)
				points.push_back(Point2d(up_y_x, up_y_y));
			
			if(std::find(a_begin, a_end, down_x_x) != a_end && std::find(b_begin, b_end, down_x_y) != b_end)
				points.push_back(Point2d(down_x_x, down_x_y));
			if(std::find(a_begin, a_end, down_y_x) != a_end && std::find(b_begin, b_end, down_y_y) != b_end)
				points.push_back(Point2d(down_y_x, down_y_y));

			if(std::find(a_begin, a_end, up_left_x) != a_end && std::find(b_begin, b_end, up_left_y) != b_end)
				points.push_back(Point2d(up_left_x, up_left_y));
			if(std::find(a_begin, a_end, up_right_x) != a_end && std::find(b_begin, b_end, up_right_y) != b_end)
				points.push_back(Point2d(up_right_x, up_right_y));

			if(std::find(a_begin, a_end, down_left_x) != a_end && std::find(b_begin, b_end, down_left_y) != b_end)
				points.push_back(Point2d(down_left_x, down_left_y));
			if(std::find(a_begin, a_end, down_right_x) != a_end && std::find(b_begin, b_end, down_right_y) != b_end)
				points.push_back(Point2d(down_right_x, down_right_y));


			}, y.arr_void());


		int64_t current_x = point.first;
		int64_t current_y = point.second;
	}
	if(radius_high == 1){return std::move(points);}
	radius_low *= radius_low;
	radius_high *= radius_high;
	radius_points_filtered(x, y, point, radius_high, radius_low, lower_x, upper_x, lower_y, upper_y, points);
	return std::move(points);

}

std::vector<Point2d > points_2d::generatePoints() const{
	/* coords.print(); */
	const Tensor& x = X_points();
	const Tensor& y = Y_points();
	std::vector<Point2d > points(x.numel());
	const int64_t* x_begin = reinterpret_cast<const int64_t*>(x.data_ptr());
	const int64_t* y_begin = reinterpret_cast<const int64_t*>(y.data_ptr());
	const int64_t* x_end = x_begin + x.numel();
	auto begin = points.begin();
	for(;x_begin != x_end; ++x_begin, ++y_begin, ++begin)
		*begin = Point2d(*x_begin, *y_begin);
	return std::move(points);

}

}
}
