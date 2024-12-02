#include "Basis.h"
#include "../Tensor.h"
#include "../utils/utils.h"
#include "Points.h"
#include <unordered_set>
#include <utility>
#include <unordered_map>
#include "KDTree.h"

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

bool BasisOverlapping::intersect(const BasisOverlapping& b2) const {
	for(const auto& Pair2 : b2.Balls){
		for(const auto& Pair : Balls){
			if(Pair.second.intersect(Pair2.second))
				return true;
		}
	}
	return false;
}

std::vector<Basis> BasisOverlapping::getConnected(const Point& p) const {
	std::vector<Basis> connections;
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


#ifdef USE_PARALLEL
uint32_t Basises::findMergeParallel(std::vector<BasisOverlapping >& ball){
	std::atomic_int64_t check;
	check.store(static_cast<int64_t>(0));
	std::atomic_uint32_t didMerge;
	didMerge.store(static_cast<uint32_t>(0));
	std::vector<BasisOverlapping > merged;
	tbb::spin_mutex mutex;
	tbb::parallel_for(tbb::blocked_range<size_t>(0, ball.size(), 20),
		[&](const tbb::blocked_range<size_t>& range){
			check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
			utils::printThreadingProgressBar(check.load(), ball.size());
			std::vector<BasisOverlapping > current;
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
uint32_t Basises::findMerge(std::vector<BasisOverlapping>& ball){
	std::vector<BasisOverlapping> merged;
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


uint32_t Basises::findMergeKDTree(std::vector<BasisOverlapping>& input_balls){
	//sorts them from biggest to smallest
	std::sort(input_balls.begin(), input_balls.end(), [](const BasisOverlapping& a, const BasisOverlapping& b){ return a.points.size() > b.points.size();});

	//make a map of the points to the balls index
	//going to use these to construct the KDTree
	std::unordered_map<Point, size_t, PointHash> pointToBasisOverlappingIndex;
	for(size_t i = 0; i < input_balls.size(); ++i){
		for(const auto& pt : input_balls[i].points){
			pointToBasisOverlappingIndex.insert({pt, i});
		}
	}

	KDTree tree(this->points.generatePoints());

	//now going to merge based on the KDTree search results
	std::vector<BasisOverlapping> merged;
	std::unordered_set<size_t> mergedIndices;
	uint32_t didMerge = 0;
	for(size_t i = 0; i < input_balls.size(); ++i){
		utils::printProgressBar(i, input_balls.size());
		if(mergedIndices.find(i) != mergedIndices.end()){continue;} //Skip if already merged
		// For each Basis in the current BasisOverlapping, search for nearby Basis using KD-Tree
		bool mergedFlag = false;
		for (const auto& pair : input_balls[i].Balls) {
			const Basis& currentBasis = pair.second;
			std::vector<Point> nearbyCenters = tree.rangeSearch(currentBasis.center, currentBasis.radius);
			for(const auto& nearbyCenter : nearbyCenters){
				size_t nearbyIndex = pointToBasisOverlappingIndex[nearbyCenter];
				if(nearbyIndex != i && (mergedIndices.find(nearbyIndex) == mergedIndices.end())){
					input_balls[i].merge(std::move(input_balls[nearbyIndex]));
					mergedIndices.insert(nearbyIndex);
					++didMerge;
					mergedFlag = true;
				}
			}
		}

		if(!mergedFlag){
			merged.push_back(std::move(input_balls[i]));
			mergedIndices.insert(i);
		}
	}
	utils::printProgressBar(input_balls.size(), input_balls.size());

	// Add remaining unmerged BasisOverlapping objects
	for (size_t i = 0; i < input_balls.size(); ++i) {
		if (mergedIndices.find(i) == mergedIndices.end()) {
			merged.push_back(std::move(input_balls[i]));
		}
	}
	input_balls = std::move(merged);
	return didMerge;
}

void Basises::radius_to(double r){
	std::vector<Point> pts = points.generatePoints();
	balls.reserve(pts.size());
	for(const auto& pt : pts){
		balls.push_back(BasisOverlapping(pt, r));
	}
	std::cout << "merging..."<<std::endl;
#ifdef USE_PARALLEL
	uint32_t didMerge = findMergeParallel(balls);
#else
	uint32_t didMerge = findMerge(balls);
#endif
	while(didMerge > 0){
		std::cout<<"did merge is "<<didMerge<<std::endl;
		didMerge = findMergeKDTree(balls);
	}
	std::cout<<"merged"<<std::endl;
	radius = r;
	std::cout << "returning"<<std::endl;
}



}} //nt::tda::
