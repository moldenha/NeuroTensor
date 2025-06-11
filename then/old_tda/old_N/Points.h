#ifndef _NT_OLD_TDA_POINTS_N_H_
#define _NT_OLD_TDA_POINTS_N_H_

#include "../Tensor.h"
#include "../utils/utils.h"
#include <_types/_uint32_t.h>
#include <array>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include <unordered_set>
#include <utility>
#include "../images/image.h"

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


// Template meta-programming to generate a tuple type containing a specified number of elements of a given type
template<typename T, size_t N, typename... REST>
struct generate_tuple_type {
    typedef typename generate_tuple_type<T, N-1, T, REST...>::type type;
};

template<typename T, typename... REST>
struct generate_tuple_type<T, 0, REST...> {
    typedef std::tuple<REST...> type;
};

template<size_t N>
using Point = typename generate_tuple_type<int64_t, N>::type;


template<size_t N, size_t Index>
inline void printHelper(std::ostream& out, const Point<N>& point){
	if constexpr (N == Index)
		return;
	out << std::get<Index>(point);
	if constexpr (Index != (N-1))
		out << ',';
	return printHelper<N, Index+1>(out, point);
}

template<size_t N>
inline std::ostream& operator << (std::ostream& out, const Point<N>& point){
	out << '(';
	printHelper<N,0>(out, point);
	return out << ')';
}


template<size_t N, size_t... Is>
inline void print_point_helper(const Point<N>& point, std::index_sequence<Is...>) {
	((std::cout << (Is == 0 ? "" : ", ") << std::get<Is>(point)), ...);
}

template<size_t N>
inline void print_point(const Point<N>& point) {
    std::cout << "(";
    print_point_helper<N>(point, std::make_index_sequence<N>());
    std::cout << ")";
}





template<size_t N, size_t... Is>
inline bool point_is_equal_helper(const Point<N>& p1, const Point<N>& p2, std::index_sequence<Is...>){
	return ((std::get<Is>(p1) == std::get<Is>(p2)) && ...);
}

template<size_t N>
inline bool operator==(const Point<N>& p1, const Point<N>& p2){
	return point_is_equal_helper(p1, p2, std::make_index_sequence<N>());
}

template<size_t N>
inline bool operator!=(const Point<N>& p1, const Point<N>& p2){return !(p1 == p2);}

template<size_t N, size_t... Is>
inline bool point_is_less_than_helper(const Point<N>& p1, const Point<N>& p2, std::index_sequence<Is...>){
	return ((std::get<Is>(p1) < std::get<Is>(p2)) && ...);
}

template<size_t N>
inline bool operator<(const Point<N>& p1, const Point<N>& p2){
	return point_is_less_than_helper(p1, p2, std::make_index_sequence<N>());
}


template<size_t N>
inline bool operator>(const Point<N>& p1, const Point<N>& p2){
	return p2 < p1;
}

template<size_t N, size_t... Is>
inline bool point_is_less_than_equal_helper(const Point<N>& p1, const Point<N>& p2, std::index_sequence<Is...>){
	return ((std::get<Is>(p1) <= std::get<Is>(p2)) && ...);
}

template<size_t N>
inline bool operator<=(const Point<N>& p1, const Point<N>& p2){
	return point_is_less_than_equal_helper(p1, p2, std::make_index_sequence<N>());
}


template<size_t N>
inline bool operator>=(const Point<N>& p1, const Point<N>& p2){
	return p2 <= p1;
}

template<size_t N, size_t... Is>
inline int64_t sum_helper(const Point<N>& p, std::index_sequence<Is...>){
	return ((std::get<Is>(p)) + ...);
}

template<size_t N>
inline int64_t sum(const Point<N>& p){
	return sum_helper<N>(p, std::make_index_sequence<N>());
}

template<size_t N, size_t... Is>
inline double point_distance_helper(const Point<N>& p1, const Point<N>& p2, std::index_sequence<Is...>) {
    int64_t sum = 0;
    // Calculate the sum of squared differences for each dimension
    ((sum += (std::get<Is>(p1) - std::get<Is>(p2)) * (std::get<Is>(p1) - std::get<Is>(p2))), ...);
    return std::sqrt(static_cast<double>(sum));
}

template<size_t N>
inline double point_distance(const Point<N>& p1, const Point<N>& p2) {
    return point_distance_helper<N>(p1, p2, std::make_index_sequence<N>());
}

template<size_t N>
struct PointNHash{
	inline std::size_t operator()(const Point<N>& p) const {

		return hash_helper(p, std::make_index_sequence<N>());
	}
	private:
	    // Helper function to hash each element of the tuple
	    template<std::size_t... Is>
	    inline std::size_t hash_helper(const Point<N>& p, std::index_sequence<Is...>) const {
		std::size_t result = 0;
		// XOR-combine the hashes of each element of the tuple
		(..., (result ^= std::hash<int64_t>{}(std::get<Is>(p))));
		return result;
	    }
};


template<size_t N>
using unordered_point_set = typename std::unordered_set<Point<N>, PointNHash<N>>;


template<size_t N>
class Points{
	Tensor original;
	Tensor coords;
	public:
		Points() = delete;
		Points(const Tensor& t, const uint8_t point);
		Points(Tensor&& t, const uint8_t point);
		unordered_point_set<N> generate_all_points_within_radius(const Point<N>& center, int64_t lower_radius, int64_t upper_radius) const; //generates all the points that make up a basis around that point
		std::vector<Point<N>> generatePoints() const; //returns a vector of all the points from the inputted tensor
};


inline Points<2> getPoints2(const std::string img_ptw, uint8_t point=1){
	images::Image img(img_ptw.c_str());
	Tensor& pixels = img.pix();
	Tensor bw = pixels[0].clone();
	bw += pixels[1];
	bw += pixels[2];
	bw /= (3.0f * 255.0f);
	/* bw.print(); */
	bw = bw.to_dtype(nt::DType::uint8);
	std::cout << "pixels shape: "<<bw.shape()<<std::endl;
	return Points<2>(std::move(bw), point);

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
