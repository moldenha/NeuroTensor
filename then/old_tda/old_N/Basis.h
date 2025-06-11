#ifndef _NT_OLD_TDA_BASIS_N_H_
#define _NT_OLD_TDA_BASIS_N_H_

#include "../Tensor.h"
#include "../utils/utils.h"
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

template<size_t N>
struct Basis{
	unordered_point_set<N> points;
	Point<N> center;
	double radius;
	explicit Basis(){}
	explicit Basis(unordered_point_set<N>&& points, Point<N> center, double r)
		:points(points), center(center), radius(r)
	{}
	inline bool intersect(const Basis& b2) const{
		double distance = point_distance<N>(center, b2.center);
		return distance <= (b2.radius + radius);
		if(b2.points.size() > points.size()){
			for(const auto& p : points){
				if(b2.points.find(p) != b2.points.end())
					return true;
			}
			return false;
		}
		for(const auto& p : b2.points){
			if(points.find(p) != points.end())
				return true;
		}
		return false;
	}
	inline void merge(unordered_point_set<N>&& ps){points.merge(std::move(ps));}
};

template<size_t N>
struct BasisOverlapping{
	std::unordered_map<Point<N>, Basis<N>, PointNHash<N> > Balls;
	unordered_point_set<N> points;
	BasisOverlapping(unordered_point_set<N>&& points, Point<N> center, double radius)
	{
		this->points.insert(center);
		points.insert(center);
		Balls[center] = Basis<N>(std::move(points), center, radius);
	}
	inline void addBasis(const Basis<N>& b){
		Balls[b.center] = b;
		points.insert(b.center);
	}
	inline void addBasis(Basis<N>&& b){
		points.insert(b.center);
		Balls[b.center] = std::move(b);
	}
	inline void addBasis(unordered_point_set<N>&& points, Point<N> center, double radius){addBasis(Basis<N>(points, center, radius));}
	inline bool intersect(const BasisOverlapping& b2){
		for(const auto& Pair2 : b2.Balls){
			for(const auto& Pair : Balls){
				if(Pair.second.intersect(Pair2.second))
					return true;
			}
		}
		return false;
	}
	inline void merge(BasisOverlapping&& b2){
		Balls.merge(b2.Balls);
		points.merge(b2.points);
	}
	inline bool contains(const Point<N>& p) const{
		return points.find(p) != points.end();
	}
	inline Basis<N>& operator[](const Point<N>& p){return Balls[p];}
	inline const Basis<N>& operator[](const Point<N>& p) const {return Balls.at(p);}
	inline bool connectedTo(const Point<N>& point, const Basis<N>& b) const {
		return b.intersect((*this)[point]);
	}
	inline std::vector<Basis<N> > getConnected(const Point<N>& p) const {
		std::vector<Basis<N> > connections;
		for(const auto& pair : Balls){
			if(pair.first == p){
				/* connections.push_back(pair.second); */
				continue;
			}
			if(connectedTo(p, pair.second))
				connections.push_back(pair.second);
		}
		return std::move(connections);
	}
};


template<size_t N>
class Basises{
	std::vector<BasisOverlapping<N> > balls;
	Points<N> points;
	double radius;
#ifdef USE_PARALLEL
	inline uint32_t findMergeParallel(std::vector<BasisOverlapping<N> >& ball){
		std::atomic_int64_t check;
		check.store(static_cast<int64_t>(0));
		std::atomic_uint32_t didMerge;
		didMerge.store(static_cast<uint32_t>(0));
		std::vector<BasisOverlapping<N> > merged;
		tbb::spin_mutex mutex;
		tbb::parallel_for(tbb::blocked_range<size_t>(0, ball.size(), 20),
			[&](const tbb::blocked_range<size_t>& range){
				check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
				utils::printThreadingProgressBar(check.load(), ball.size());
				std::vector<BasisOverlapping<N> > current;
				current.reserve(range.end() - range.begin());
				current.push_back(std::move(ball[range.begin()]));
				uint32_t current_didMerge = 1;
				for(uint32_t i = range.begin()+1; i < range.end(); ++i){
					bool finished_merge = false;
					for(auto& existing : current){
						if(existing.intersect(ball[i])){
							existing.merge(std::move(ball[i]));
							++current_didMerge;
							finished_merge = true;
							break;
						}
					}
					if(!finished_merge){current.push_back(std::move(ball[i]));}
				}

				tbb::spin_mutex::scoped_lock lock(mutex);
				merged.reserve(current.size()+1);
				merged.insert(merged.end(), current.begin(), current.end());
				lock.release();
				didMerge.fetch_add(current_didMerge, std::memory_order_relaxed);
			});
		ball = std::move(merged);
		return didMerge.load();

	}
#endif
	inline uint32_t findMerge(std::vector<BasisOverlapping<N> >& ball){
		std::vector<BasisOverlapping<N> > merged;
		merged.reserve(ball.size());
		merged.push_back(std::move(ball[0]));
		uint32_t didMerge = 0;
		for(uint32_t i = 1; i < ball.size(); ++i){
			utils::printProgressBar(i, ball.size());
			bool finished_merged = false;
			for(auto& existing : merged){
				if(existing.intersect(ball[i])){
					existing.merge(std::move(ball[i]));
					++didMerge;
					finished_merged = true;
					break;
				}
			}
			if(!finished_merged){merged.push_back(std::move(ball[i]));}
		}
		ball = std::move(merged);
		return didMerge;
	}
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
		explicit Basises(Points<N> point)
			:points(point),
			radius(0)
		{}
		inline void radius_to(double r, bool get_inner_points=true){
			std::vector<Point<N> > pts = points.generatePoints();
			balls.reserve(pts.size());

#ifdef USE_PARALLEL
			if(get_inner_points){
			tbb::spin_mutex mutex;
			std::atomic_int64_t check;
			check.store(static_cast<int64_t>(0));
			tbb::parallel_for(tbb::blocked_range<size_t>(0, pts.size()),
					[&](const tbb::blocked_range<size_t>& range){
				check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
				utils::printThreadingProgressBar(check.load(), pts.size());
				std::vector<BasisOverlapping<N>> curBalls;
				curBalls.reserve(range.end()-range.begin());
				auto begin = pts.cbegin() + range.begin();
				auto end = pts.cbegin() + range.end();
				for(;begin != end; ++begin)
					curBalls.emplace_back(BasisOverlapping<N>(points.generate_all_points_within_radius(*begin, 0, r), *begin, r));

				tbb::spin_mutex::scoped_lock lock(mutex);
				balls.insert(balls.end(), curBalls.begin(), curBalls.end());
				lock.release();
			});
			}else{
#else
			if(get_inner_points){
			uint32_t counter = 0;
			for(const auto& pt : pts){
				utils::printProgressBar(counter, pts.size());
				balls.push_back(BasisOverlapping<N>(points.generate_all_points_within_radius(pt, 0, r), pt, r));
				++counter;
			}
			}else{
#endif
				uint32_t counter = 0;
				for(const auto& pt : pts){
					utils::printProgressBar(counter, pts.size());
					balls.push_back(BasisOverlapping<N>(unordered_point_set<N>({pt}), pt, r));
					++counter;
				}	
			}
			std::cout << "merging..."<<std::endl;
#ifdef USE_PARALLEL
			uint32_t didMerge = findMergeParallel(balls);
#else
			uint32_t didMerge = findMerge(balls);
#endif
			while(didMerge > 0){
				std::cout<<"did merge is "<<didMerge<<std::endl;
				didMerge = findMerge(balls);
			}
			std::cout<<"merged"<<std::endl;
			radius = r;
			std::cout << "retugning"<<std::endl;
		}
		inline const int64_t& getRadius() const {return radius;}
		inline const std::vector<BasisOverlapping<N> > getBalls() const {return balls;}
		inline const BasisOverlapping<N>& operator[](const Point<N>& p){
			/* utils::throw_exception(points.find(p) != points.end(), "This collection of $ overlapping basises does not have the point", balls.size()); */
			for(const auto& ball : balls){
				if(ball.contains(p)){return ball;}
			}
			utils::throw_exception(false, "Erorr, point not found in basises");
			return balls.back();
		}
		inline const Basis<N>& getBasis(const Point<N>& p){return (*this)[p][p];}
};


}
}

#endif
