#include "BatchBasis.h"
#include "../../Tensor.h"
#include "../../utils/utils.h"
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
#include <algorithm>
#include <functional>

namespace nt{
namespace tda{

/* bool BatchBasisOverlapping::intersects(const BasisBatchOverlapping& bs, int64_t dim) const{ */
/* 	for(const auto& Pair2 : b2.Balls){ */
/* 		if(Pair2.first.first != dim){continue;} */
/* 		for(const auto& Pair : Balls){ */
/* 			if(Pair.first.first != dim){continue;} */
/* 			if(Pair.second.intersects(Pair2.second)) */
/* 				return true; */
/* 		} */
/* 	} */
/* 	return false; */
/* } */

/* void BatchBasisOverlapping::merge_batch(BasisBatchOverlapping& bs, int64_t batch){ */
/* 	//first the points */
/* 	auto begin = bs.points.begin(); */
/* 	auto end = bs.points.end(); */
/* 	auto batch_begin = begin; */
/* 	auto batch_end = end; */
/* 	bool on_batch = false; */
/* 	for(;begin != end; ++begin){ */
/* 		if(begin->first == batch){ */
/* 			if(on_batch){continue;} */
/* 			batch_begin = begin; */
/* 			on_batch = true; */
/* 		}else if(on_batch){ */
/* 			batch_end = begin; */
/* 			on_batch = false; */
/* 			this->points.insert(batch_begin, batch_end); */
/* 			bs.points.erase(batch_begin, batch_end); */
/* 		} */
/* 	} */
/* 	if(on_batch){ */
/* 		this->points.insert(batch_begin, batch_end); */
/* 		bs.points.erase(batch_begin, batch_end); */
/* 	} */

/* 	//next the map */
/* 	on_batch = false; */
/* 	auto m_begin = bs.Balls.begin(); */
/* 	auto m_end = bs.Balls.end(); */
/* 	auto m_batch_begin = m_begin; */
/* 	auto m_batch_end = m_end; */
/* 	for(;m_begin != m_end; ++m_begin){ */
/* 		if(m_begin->first->first == batch){ */
/* 			if(on_batch){continue;} */
/* 			m_batch_begin = m_begin; */
/* 			on_batch = true; */
/* 		}else if(on_batch){ */
/* 			m_batch_end = m_begin; */
/* 			on_batch = false; */
/* 			this->Balls.insert(m_batch_begin, m_batch_end); */
/* 			bs.Balls.erase(m_batch_begin, m_batch_end); */
/* 		} */
/* 	} */
/* 	if(on_batch){ */
/* 		this->Balls.insert(m_batch_begin, m_batch_end); */
/* 		bs.Balls.erase(m_batch_begin, m_batch_end); */
/* 	} */
/* } */



void BatchBasises::generateTracking(const BatchPoints& pts){
	const int64_t batches = pts.batches();
	int64_t total_points = 0;

	balls.reserve(batches);
	// Pre-allocate memory in pointToBasisOverlappingIndex
	int64_t total = pts.count_points();
	std::cout << "a total of "<<total<<" points"<<std::endl;
	pointToBasisOverlappingIndex.reserve(total);

	for(int64_t b = 0; b < batches; ++b){
		//itterate through each batch
		const std::vector<Point>& b_pts = pts.generatePoints(b);//get the points for that batch index
		utils::printProgressBar(b, batches, " generating "+std::to_string(b_pts.size())+" points");
		this->balls.push_back(std::vector<BasisOverlapping>());
		std::vector<BasisOverlapping>& vec = this->balls[b];
		vec.reserve(b_pts.size());
		/* this->pointToBasisOverlappingIndex.reserve(b_pts.size()); */
		for(size_t balls_index = 0; balls_index < b_pts.size(); ++balls_index){
			vec.emplace_back(b_pts[balls_index], this->radius);
			this->pointToBasisOverlappingIndex.insert({{b, b_pts[balls_index]}, balls_index});
			
		}
		this->tree.build_batch(const_cast<std::vector<Point>&>(b_pts), b);
	}
	utils::printProgressBar(batches, batches, " generated tracking");
}


void BatchBasises::sortBalls(){
	const int64_t batches = this->balls.size();
	for(int64_t i = 0; i < batches; ++i){
		std::vector<BasisOverlapping>& input_balls = this->balls[i];
		std::sort(input_balls.begin(), input_balls.end(), 
				[](const BasisOverlapping& a, const BasisOverlapping& b)
				{ return a.points.size() > b.points.size();});
		for(size_t j = 0; j < input_balls.size(); ++j){
			for(const auto& pt : input_balls[j].points){
				this->pointToBasisOverlappingIndex[std::pair<int64_t, Point>(i, pt)] = j;
			}
		}
	}
}

void BatchBasises::sortBalls(std::vector<uint32_t>& prevMerged){
	const int64_t batches = this->balls.size();
	for(int64_t i = 0; i < batches; ++i){
		if(prevMerged[i] == 0){continue;}
		std::vector<BasisOverlapping>& input_balls = this->balls[i];
		std::sort(input_balls.begin(), input_balls.end(), 
				[](const BasisOverlapping& a, const BasisOverlapping& b)
				{ return a.points.size() > b.points.size();});
		for(size_t j = 0; j < input_balls.size(); ++j){
			for(const auto& pt : input_balls[j].points){
				this->pointToBasisOverlappingIndex[std::pair<int64_t, Point>(i, pt)] = j;
			}
		}
		/* std::cout << "this balls size is: "<<this->balls[i].size()<<std::endl; */
	}
}

template <typename K, typename V, typename Hash>
bool can_merge(const std::unordered_map<K, V, Hash>& map1, const std::unordered_map<K, V, Hash>& map2) {
    for (const auto& [key, value] : map2) {
        if (map1.find(key) != map1.end()) {
            return false; // Conflict found
        }
    }
    return true; // No conflicts
}

void BatchBasises::findMergeKDTree(std::vector<uint32_t>& prevMerged, bool verbose){
	const int64_t batches = this->balls.size();
	std::vector<std::reference_wrapper<const Point>> nearbyCenters;
	nearbyCenters.reserve(30);
	std::unordered_set<size_t> mergedIndices;
	mergedIndices.reserve(*std::max_element(prevMerged.cbegin(), prevMerged.cend()));
	for(int64_t b = 0; b < batches; ++b){
        if(verbose){
	    	utils::printProgressBar(b, batches);
		    std::cout << "\033[F";
        }
		if(prevMerged[b] == 0){continue;}
		std::unordered_set<size_t> mergedIndices;
		prevMerged[b] = 0; //the didMerge variable
		std::vector<BasisOverlapping>& input_balls = this->balls[b];
		if(input_balls.size() == 1){continue;}
		for(size_t i = 0; i < input_balls.size(); ++i){
			if(verbose){utils::printProgressBar(i, input_balls.size());}
			//skip indices already merged
			if(mergedIndices.find(i) != mergedIndices.end()){continue;}
			bool mergedFlag = false;
			for(const auto& pair : input_balls[i].Balls){
				const Basis& currentBasis = pair.second;
				tree.rangeSearch(b, currentBasis.center, currentBasis.radius, nearbyCenters);
				for(const auto& nearbyCenter_ref : nearbyCenters){
					const Point& nearbyCenter = nearbyCenter_ref.get();
					const size_t& nearbyIndex = this->pointToBasisOverlappingIndex[std::pair<int64_t, Point>(b, nearbyCenter)];
					if(nearbyIndex != i && (mergedIndices.find(nearbyIndex) == mergedIndices.end())){
						/* std::cout << "merging "<<i<<" and "<<nearbyIndex<<" input ball size: "<<input_balls.size()<<std::endl; */
						input_balls[i].merge(std::move(input_balls[nearbyIndex]));
						mergedIndices.insert(nearbyIndex);
						++prevMerged[b];
						mergedFlag = true;
					}
				}
				nearbyCenters.clear(); //clear the nearby centers, so that more memory is not allocated
			}
		}
		// Remove all merged BasisOverlapping objects
		/* std::cout << "indexes merged: "<<mergedIndices.size() << std::endl; */
		auto newEnd = std::remove_if(input_balls.begin(), input_balls.end(),
		    [&mergedIndices, &input_balls](const BasisOverlapping& basis) {
			size_t index = &basis - &input_balls[0];
			return mergedIndices.find(index) != mergedIndices.end();
		    }
		);
		input_balls.erase(newEnd, input_balls.end());
		input_balls.shrink_to_fit();
		mergedIndices.clear();//clear the merged indices so that more memory is not allocated

		if(verbose){std::cout << "\n";}

		// Add remaining unmerged BasisOverlapping objects
		/* for (size_t i = 0; i < input_balls.size(); ++i) { */
		/* 	if (mergedIndices.find(i) == mergedIndices.end()) { */
		/* 		merged.push_back(std::move(input_balls[i])); */
		/* 	} */
		/* } */
		/* this->balls[b] = std::move(merged); */
	}
	/* std::cout << "broke loop, now sorting..."<<std::endl; */

	sortBalls(prevMerged); //this is called at the end of this function to ensure the balls are always sorted
}




void BatchBasises::radius_to(double r, bool verbose){
	for(size_t b = 0; b < this->balls.size(); ++b){
		for(size_t i = 0; i < this->balls[b].size(); ++i){
			balls[b][i].adjust_radius(r);
		}
	}
	this->radius = r;
    if(verbose){
        std::cout << "merging..." << std::endl;
        std::vector<uint32_t> prevMerged(this->balls.size(), 1);
        utils::beginDualProgressBar(this->balls.size(), this->balls[0].size());
        size_t num = 0;
        while(num < prevMerged.size()){
            this->findMergeKDTree(prevMerged, true);
            num = std::count(prevMerged.begin(), prevMerged.end(), 0);
            utils::endDualProgressBar(this->balls.size(), this->balls[0].size());
            std::cout << std::endl << num << " have finished merging"<<std::endl;
            utils::beginDualProgressBar(this->balls.size(), this->balls[0].size());
        }
    }else{
        std::vector<uint32_t> prevMerged(this->balls.size(), 1);
        size_t num = 0;
        while(num < prevMerged.size()){
            this->findMergeKDTree(prevMerged, false);
            num = std::count(prevMerged.begin(), prevMerged.end(), 0);
        }
    }
}

}} //nt::tda::


