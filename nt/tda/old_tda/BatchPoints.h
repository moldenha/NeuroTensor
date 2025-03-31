#ifndef _NT_OLD_TDA_BATCH_POINTS_H_
#define _NT_OLD_TDA_BATCH_POINTS_H_

#include "../../Tensor.h"
#include "../../utils/utils.h"
#include <_types/_uint32_t.h>
#include <array>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include <unordered_set>
#include <utility>
#include "../../images/image.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "Points.h"

namespace nt{
namespace tda{


class BatchPoints{
	Tensor original;
	/* Tensor coords; */
	std::vector<Point> pts;
	/* KDTree tree; */
	const int64_t dim;
	uint8_t point;
	int64_t last;
	public:
		BatchPoints() = delete;
		BatchPoints(const Tensor& t, const uint8_t point);
		BatchPoints(Tensor&& t, const uint8_t point);
		BatchPoints(const nt::Tensor& t, const uint8_t point, const int64_t dim); //point is the number that it is looking for
		BatchPoints(nt::Tensor&& t, const uint8_t point, const int64_t dim); //point is the number that it is looking for
		void generate_all_points_within_radius(const Point& center, double lower_radius, double upper_radius, const int64_t index, unordered_point_set& points) const; //generates all the points that make up a basis around that point

		const std::vector<Point>& generatePoints(const int64_t batch) const; //returns a vector of all the points from the inputted tensor
		inline const int64_t& dims() const noexcept {return dim;}
		inline const int64_t& batches() const noexcept {return original.shape()[0];}
		const int64_t count_points() const noexcept;
};


}} //nt::tda::



#endif
