#ifndef _NT_OLD_TDA_BASIS_H_
#define _NT_OLD_TDA_BASIS_H_

#include "../../Tensor.h"
#include "../../utils/utils.h"
#include "Points.h"
#include <unordered_set>
#include <utility>
#include <unordered_map>

#ifdef USE_PARALLEL
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#endif
#include <atomic>

namespace nt{
namespace tda{

struct Basis{
	/* unordered_point_set points; */
	Point center;
	double radius;
	explicit Basis(Point center, double r)
		:center(center), radius(r)
	{}
	Basis(Basis&& B)
		:center(std::move(B.center)), radius(std::exchange(B.radius, 0))
	{}
	Basis(const Basis& B)
		:center(B.center), radius(B.radius)
	{}
	Basis()
		:center(0), radius(0)
	{}
	inline Basis& operator=(Basis&& b){
		center = std::move(b.center);
		radius = std::exchange(b.radius, 0);
		return *this;
	}
	inline Basis& operator=(const Basis& b){
		center = b.center;
		radius = b.radius;
		return *this;
	}
	inline bool intersect(const Basis& b2) const{
		double distance = point_distance(center, b2.center);
		return distance <= (b2.radius + radius);
	}
	inline const int64_t dims() const noexcept {return center.size();}
};

struct BasisOverlapping{
	std::unordered_map<Point, Basis, PointHash > Balls;
	unordered_point_set points;
	const int64_t dim;
	BasisOverlapping(Point center, double radius)
		:dim(center.size())
	{
		this->points.insert(center);
		points.insert(center);
		Balls[center] = std::move(Basis(center, radius));
	}
	BasisOverlapping(const BasisOverlapping& other)
		:Balls(other.Balls),
		points(other.points),
		dim(other.dim)
	{}
	inline BasisOverlapping& operator=(const BasisOverlapping& other){
		Balls = other.Balls;
		points = other.points;
		const_cast<int64_t&>(dim) = other.dim;
		return *this;
	}
	inline BasisOverlapping& operator=(BasisOverlapping&& other){
		Balls = std::move(other.Balls);
		points = std::move(other.points);
		const_cast<int64_t&>(dim) = other.dim;
		const_cast<int64_t&>(other.dim) = 0;
		return *this;
	}
	inline void addBasis(const Basis& b){
		Balls[b.center] = b;
		points.insert(b.center);
	}
	inline void addBasis(Basis&& b){
		points.insert(b.center);
		Balls[b.center] = std::move(b);
	}
	inline void addBasis(Point center, double radius){addBasis(Basis(center, radius));}
	bool intersect(const BasisOverlapping& b2) const;
	inline void merge(BasisOverlapping&& b2){
		Balls.merge(b2.Balls);
		points.merge(b2.points);
	}
	inline bool contains(const Point& p) const{
		return points.find(p) != points.end();
	}
	inline Basis& operator[](const Point& p){return Balls[p];}
	inline const Basis& operator[](const Point& p) const {return Balls.at(p);}
	inline bool connectedTo(const Point& point, const Basis& b) const {
		return b.intersect((*this)[point]);
	}
	std::vector<Basis> getConnected(const Point& p) const;
	inline const int64_t& dims() const noexcept {return dim;}
	inline void adjust_radius(double r){
		for(auto& Pair : Balls){
			Pair.second.radius = r;
		}
	}
	
};


class Basises{
	std::vector<BasisOverlapping> balls;
	Points points;
	double radius;
#ifdef USE_PARALLEL
	uint32_t findMergeParallel(std::vector<BasisOverlapping >& ball);
#endif
	uint32_t findMerge(std::vector<BasisOverlapping>& ball);
	uint32_t findMergeKDTree(std::vector<BasisOverlapping>& ball);
	public:
		Basises() = delete;
		explicit Basises(const Tensor& t, uint8_t point)
			:points(t, point),
			radius(0)
		{}
		explicit Basises(Tensor&& t, uint8_t point)
			:points(std::move(t), point),
			radius(0)
		{}
		explicit Basises(Points point)
			:points(point),
			radius(0)
		{}
		void radius_to(double r);
		inline void adjust_radius(double r){ //this should be used after radius_to
			for(auto& clump : balls){
				clump.adjust_radius(r);
			}
			uint32_t didMerge = findMerge(balls);
			while(didMerge > 0){
				didMerge = findMerge(balls);
			}
			radius = r;
		}
		inline const double& getRadius() const noexcept {return radius;}
		inline const std::vector<BasisOverlapping>& getBalls() const noexcept {return balls;}
		inline const BasisOverlapping& operator[](const Point& p) const {
			/* utils::throw_exception(points.find(p) != points.end(), "This collection of $ overlapping basises does not have the point", balls.size()); */
			for(const auto& ball : balls){
				if(ball.contains(p)){return ball;}
			}
			utils::throw_exception(false, "Erorr, point not found in basises");
			return balls.back();
		}
		inline const Basis& getBasis(const Point& p){return (*this)[p][p];}
		inline const int64_t& dims() const noexcept {return points.dims();}
};


}
}

#endif //_NT_OLD_TDA_BASIS_H_
