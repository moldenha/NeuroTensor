#ifndef _NT_TDA_BATCH_BASIS_H_
#define _NT_TDA_BATCH_BASIS_H_

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

#include "Basis.h"
#include "BatchPoints.h"
#include "BatchKDTree.h"

namespace nt{
namespace tda{

//going to keep the Basis struct the same
//it just holds the point and the radius associated with it


struct BatchPointHash{
	// Hash function for Point
	std::size_t operator()(const std::pair<int64_t, Point>& p) const {
		std::size_t hash = 0;
		hash ^= std::hash<int64_t>{}(p.first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
		for (const auto& value : p.second) {
			// Combine each value's hash with the overall hash
			hash ^= std::hash<int64_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
		}
		return hash;
	}	
};




//going to make this a map
//where it is going to be basically the same as BasisOverlapping
//except to access a specific batch, there is going to be a map pointing to it's specific basis's points
/* struct BatchBasisOverlapping{ */
/* 	//it will hold all of them */
/* 	//this will help with any memory fragmentation */
/* 	std::map<int64_t, BasisOverlapping> overlaps; //this is a map from each batch to the basis overlap */
/* 	BatchBasisOverlapping() = default; */
/* 	BatchBasisOverlapping(Point center, double radius, int64_t batch){ */
/* 		overlaps[batch] = BasisOverlap(center, radius); */
/* 	} */
/* 	inline void addOverlap(const BasisOverlap& b, int64_t batch){ */
/* 		utils::THROW_EXCEPTION(overlaps.count(batch) == 0, "Already have overlapping basis's occupying that batch"); */
/* 		overlaps[batch] = b; */
/* 	} */
/* 	inline void addOverlap(const BasisOverlap& b, int64_t batch){ */
/* 		utils::THROW_EXCEPTION(overlaps.count(batch) == 0, "Already have overlapping basis's occupying that batch"); */
/* 		overlaps[batch] = b; */
/* 	} */
/* 	inline void addOverlap(Point center, double radius, int64_t batch){ */
/* 		utils::THROW_EXCEPTION(overlaps.count(batch) == 0, "Already have overlapping basis's occupying that batch"); */
/* 		overlaps[batch] = BasisOverlap(center, radius); */
/* 	} */
/* 	inline BasisOverlapping& operator[](int64_t batch){return overlaps.at(batch);} */
/* 	inline const BasisOverlapping& operator[](int64_t batch) const {return overlaps.at(batch);} */
/* 	inline void adjustRadius(double r){ */
/* 		for(auto& pair : overlaps){ */
/* 			pair.second.adjustRadius(r); */
/* 		} */
/* 	} */
/* }; */


class BatchBasises{
	std::vector<std::vector<BasisOverlapping>> balls; //each vector represent a batch, then a vector of balls
	double radius;
	BatchKDTree tree;
	std::unordered_map<std::pair<int64_t, Point>, size_t, BatchPointHash> pointToBasisOverlappingIndex;
	void findMergeKDTree(std::vector<uint32_t>&);
	void generateTracking(const BatchPoints& pts);
	void sortBalls();
	void sortBalls(std::vector<uint32_t>&);
	public:
		BatchBasises(const BatchPoints& pts)
			:radius(0),
			tree(pts.dims(), pts.batches())
		{generateTracking(pts);}
		void radius_to(double r);
		inline const std::vector<BasisOverlapping>& get_balls(int64_t batch){return balls[batch];}
		//it is zero because it is sorted after every merge
		inline const BasisOverlapping& getLargest(int64_t batch) const {return balls[batch][0];}
		inline const BasisOverlapping& getBasisOverlapping(int64_t batch, const Point& p){
			return balls[batch][pointToBasisOverlappingIndex[std::pair<int64_t, Point>(batch, p)]];
		}
};

}}

#endif
