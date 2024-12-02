#ifndef TDA_POINTS_H
#define TDA_POINTS_H

//forward declarations
namespace nt{namespace tda{
class Point;
class Points;
}}


#include "../Tensor.h"
#include "../utils/utils.h"
#include <_types/_uint32_t.h>
#include <array>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include <unordered_set>
#include <utility>
#include "../images/image.h"
#include "../intrusive_ptr/intrusive_ptr.hpp"
#include "../refs/intrusive_list.h"
/* #include "KDTree.h" */

namespace nt{
namespace tda{

/* struct point_2d{ */
/* 	std::array<int64_t, 2> points; //(x, y) */
/* 	inline const int64_t& x() const {return points[0];} */
/* 	inline int64_t& x() {return points[0];} */
/* 	inline const int64_t& y() const {return points[1];} */
/* 	inline int64_t& y() {return points[1];} */
/* 	point_2d(const int64_t x, const int64_t y) */
/* 		:points({x, y}) */
/* 	{} */
/* }; */

using Point2d = typename std::pair<int64_t, int64_t>;

struct Point2dHash{
	inline std::size_t operator()(const Point2d& p) const {
		return std::hash<int64_t>{}(p.first) ^ std::hash<int64_t>{}(p.second);
	}
};

struct Point2dEqual {
    inline bool operator()(const Point2d& p1, const Point2d& p2) const {
        return p1.first == p2.first && p1.second == p2.second;
    }
};

using unordered_point2d_set = typename std::unordered_set<Point2d, Point2dHash>;

class points_2d{
	Tensor original;
	Tensor coords;
	public:
		points_2d() = delete;
		points_2d(const nt::Tensor& t, const uint8_t point); //point is the number that it is looking for
		points_2d(nt::Tensor&& t, const uint8_t point); //point is the number that it is looking for
		Tensor in_radius(Point2d, int64_t) const;
		std::vector<Point2d > in_radius_vec(const Point2d&, int64_t) const;
		std::vector<Point2d > in_radius_vec(const Point2d&, int64_t, int64_t) const;
		std::vector<Point2d > in_radius_filtered(const Point2d&, int64_t, int64_t) const;
		unordered_point2d_set in_radius_one(const Point2d&) const;
		inline const Tensor& X_points() const {return reinterpret_cast<const Tensor*>(coords.data_ptr())[0];}
		inline const Tensor& Y_points() const {return reinterpret_cast<const Tensor*>(coords.data_ptr())[1];}
		std::vector<Point2d> generatePoints() const;
		inline Tensor& original_tensor() {return original;}
		inline const Tensor& original_tensor() const {return original;}
		
	
};



class Point{
	intrusive_ptr<intrusive_list<int64_t>> ptr;
	public:
		Point()
			:ptr(make_intrusive<intrusive_list<int64_t> >(0))
		{}
		explicit Point(int64_t n)
			:ptr(make_intrusive<intrusive_list<int64_t> >(n))
		{}
		explicit Point(std::initializer_list<int64_t> l)
			:ptr(make_intrusive<intrusive_list<int64_t> >(l.size()))
		{
			std::copy(l.begin(), l.end(), begin());
		}
		Point(Point&& p)
			:ptr(std::move(p.ptr))
		{}
		Point(const Point& p)
			:ptr(make_intrusive<intrusive_list<int64_t> >(p.size()))
		{
			std::copy(p.cbegin(), p.cend(), begin());
		}
		Point& operator=(const Point& p){
			ptr = make_intrusive<intrusive_list<int64_t> >(p.size());
			std::copy(p.cbegin(), p.cend(), ptr->ptr());
			return *this;
		}
		Point& operator=(Point&& p){
			ptr = std::move(p.ptr);
			return *this;
		}
		//explicit function to share the memory
		inline Point& share(const Point& p) noexcept{
			ptr = p.ptr;
			return *this;
		}
		inline int64_t& operator[](int64_t n) noexcept {return ptr->at(n);}
		inline const int64_t& operator[](int64_t n) const noexcept {return ptr->at(n);}
		inline const int64_t& size() const noexcept {return ptr->size();}
		inline const int64_t& dims() const noexcept {return ptr->size();}
		inline int64_t* begin() noexcept {return ptr->ptr();}
		inline int64_t* end() noexcept {return ptr->end();}
		inline const int64_t* begin() const noexcept {return ptr->ptr();}
		inline const int64_t* end() const noexcept {return ptr->end();}
		inline const int64_t* cbegin() const noexcept {return ptr->ptr();}
		inline const int64_t* cend() const noexcept {return ptr->end();}
		inline Point clone() const noexcept {
			Point out(size());
			std::copy(cbegin(), cend(), out.begin());
			return std::move(out);
		}
		inline Point operator+(int64_t element) const noexcept{
			Point out = clone();
			for(auto& ele : out)
				ele += element;
			return std::move(out);
		}
		inline Point& operator+=(int64_t element) noexcept{
			for(auto& ele : *this)
				ele += element;
			return *this;
		}
		inline const int64_t& back() const noexcept {return (*this)[size()-1];}
		inline int64_t& back() noexcept {return (*this)[size()-1];}
		inline const bool operator==(const Point& p) const noexcept{
			if(p.size() != size()){return false;}
			auto p_b = p.begin();
			for(auto begin = cbegin(); begin != cend(); ++begin, ++p_b)
				if(*begin != *p_b){return false;}
			return true;
		}
		inline const bool operator!=(const Point& p) const noexcept{return !((*this) == p);}


};




inline int64_t sum(const Point& p) noexcept {
	return std::accumulate(p.cbegin(), p.cend(), int64_t(0));
}

inline std::ostream& operator<<(std::ostream& os, const Point& p){
	os << '(';
	for(size_t i = 0; i < p.size()-1; ++i)
		os << p[i]<<',';
	return os << p.back() << ')';
}




inline double point_distance(const Point& p1, const Point& p2) {
	utils::throw_exception(p1.size() == p2.size(), "Expected to get points of same dimension but got $d and $d", p1.size(), p2.size());
	int64_t sum = 0;
	for(size_t i = 0; i < p1.size(); ++i)
		sum += ((p1[i] - p2[i]) * (p1[i] - p2[i]));
	return std::sqrt(static_cast<double>(sum));
}

struct PointHash{
	// Hash function for Point
	std::size_t operator()(const Point& p) const {
		std::size_t hash = 0;
		for (const auto& value : p) {
			// Combine each value's hash with the overall hash
			hash ^= std::hash<int64_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
		}
		return hash;
	}	
};


using unordered_point_set = typename std::unordered_set<Point, PointHash>;


class Points{
	Tensor original;
	Tensor coords;
	std::vector<Point> pts;
	/* KDTree tree; */
	const int64_t dim;
	public:
		Points() = delete;
		Points(const Tensor& t, const uint8_t point);
		Points(Tensor&& t, const uint8_t point);
		Points(const nt::Tensor& t, const uint8_t point, const int64_t dim); //point is the number that it is looking for
		Points(nt::Tensor&& t, const uint8_t point, const int64_t dim); //point is the number that it is looking for
		unordered_point_set generate_all_points_within_radius(const Point& center, double lower_radius, double upper_radius) const; //generates all the points that make up a basis around that point
		std::vector<Point> generatePoints() const; //returns a vector of all the points from the inputted tensor
		inline const int64_t& dims() const noexcept {return dim;}
		Points& reset(const Tensor&, const uint8_t);
		Points& reset(const Tensor&, const uint8_t, const int64_t);
		
};


inline Points getPoints2(const std::string img_ptw, uint8_t point=1){
	images::Image img(img_ptw.c_str());
	Tensor& pixels = img.pix();
	Tensor bw = pixels[0].clone();
	bw += pixels[1];
	bw += pixels[2];
	bw /= (3.0f * 255.0f);
	/* bw.print(); */
	bw = bw.to_dtype(nt::DType::uint8);
	std::cout << "pixels shape: "<<bw.shape()<<std::endl;
	return Points(std::move(bw), point, 2);

}

inline points_2d getPoints2d(const std::string img_ptw, uint8_t point=1){
	images::Image img(img_ptw.c_str());
	Tensor& pixels = img.pix();
	Tensor bw = pixels[0].clone();
	bw += pixels[1];
	bw += pixels[2];
	bw /= (3.0f * 255.0f);
	/* bw.print(); */
	bw = bw.to_dtype(nt::DType::uint8);
	std::cout << "pixels shape: "<<bw.shape()<<std::endl;
	return points_2d(std::move(bw), point);
}

}

}

#endif
