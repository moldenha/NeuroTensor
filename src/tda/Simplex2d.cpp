#include "../Tensor.h"
#include "../utils/utils.h"
#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <array>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include "Points.h"
#include "Simplex.h"
#include <cmath>
#include <utility>
#include <unordered_set>
#ifdef USE_PARALLEL
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>
#include <tbb/mutex.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>
#endif
#include <atomic>

namespace nt{
namespace tda{


/* void utils::printProgressBar(uint32_t progress, uint32_t total, uint32_t width = 50) { */
/*     float percentage = static_cast<float>(progress) / total; */
/*     int numChars = static_cast<int>(percentage * width); */

/*     std::cout << "\r["; */
/*     for (int i = 0; i < numChars; ++i) { */
/*         std::cout << "="; */
/*     } */
/*     for (int i = numChars; i < width; ++i) { */
/*         std::cout << " "; */
/*     } */
/*     std::cout << "] " << static_cast<int>(percentage * 100.0) << "% "<<progress<<'/'<<total; */
/*     std::cout.flush(); */
/* } */

inline bool in_radius(const Point2d& a, const Point2d& b, const int64_t& radius){
	if( (((a.first - b.first) * (a.first - b.first)) + ((a.second - b.second) * (a.second - b.second))) <= radius)
		return true;
	return false;
}

inline bool already_simplex(const Simplex2d& nSimplex, const std::vector<Simplex2d >& simplexes){
	auto cbegin = simplexes.cbegin();
	auto cend = simplexes.cend();
	for(;cbegin != cend; ++cbegin){
		bool found = false;
		for(uint32_t i =0; i < 3; ++i){
			if((*cbegin)[i].first == nSimplex[0].first && (*cbegin)[i].second == nSimplex[0].second){
				found = true;
				break;
			}
		}
		if(!found){continue;}
		found = false;
		for(uint32_t i =0; i < 3; ++i){
			if((*cbegin)[i].first == nSimplex[1].first && (*cbegin)[i].second == nSimplex[1].second){
				found = true;
				break;
			}
		}
		if(!found){continue;}
		found = false;
		for(uint32_t i =0; i < 3; ++i){
			if((*cbegin)[i].first == nSimplex[2].first && (*cbegin)[i].second == nSimplex[2].second){
				found = true;
				break;
			}
		}
		if(!found){continue;}
		return true;
	}
	return false;
}



// Comparator function to compare points based on y-coordinate (descending order)
bool comparePoints(const Point2d& p1, const Point2d& p2) {
    return (p1.first + p1.second) > (p2.first + p2.second);
}


inline bool equal(const Point2d& a, const Point2d& b){
	return a.first == b.first && a.second == b.second;
}

inline bool equal(const Point2d& a, const Point2d& b, const Point2d& c){
	return equal(a,b) || equal(b,c) || equal(c,a);
}

simplexes_2d::simplexes_2d(const points_2d& points, int64_t radius){
	std::vector<Point2d > all_points = points.generatePoints();
	int64_t rSquared = radius * radius;
	auto cbegin = all_points.cbegin();
	auto cend = all_points.cend();
	std::cout << "have "<<all_points.size() << " to check"<<std::endl;
	int64_t checked = 0;
	std::vector<std::vector<Point2d > > all_radi(all_points.size());
	for(;cbegin != cend; ++cbegin, ++checked){
		utils::printProgressBar(checked, all_points.size());
		/* std::cout << checked<<"/"<<all_points.size()<<std::endl; */
		all_radi[checked] = points.in_radius_vec(*cbegin, radius);
		
	}
	std::cout << std::endl;
	cbegin = all_points.cbegin();
	auto begin = all_radi.begin();
	checked = 0;
	unordered_simplex2d_set set;
	/* unordered_point2d_set set; */
#ifdef USE_PARALLEL
	tbb::spin_mutex mutex;
	tbb::blocked_range<size_t> b(0, all_points.size());
	std::cout<<"b grain size:"<<b.grainsize()<<", size: "<<b.size()<<std::endl;
	tbb::parallel_for(b,
			[&](const tbb::blocked_range<size_t>& range) {
			std::cout << "doing "<<range.begin()<<" to "<<range.end()<<std::endl;
			auto m_begin = all_radi.cbegin() + range.begin();
			auto m_end = all_radi.cbegin() + range.end();
			auto cm_begin = all_points.cbegin() + range.begin();
			unordered_simplex2d_set mySet;
			for(;m_begin != m_end; ++m_begin, ++cm_begin){
				for(uint32_t i = 0; i < m_begin->size(); ++i){
					for(uint32_t j = i+1; j < m_begin->size(); ++j){
						if(equal(*cm_begin, (*m_begin)[i], (*m_begin)[j]))
							continue;
						if(!in_radius((*m_begin)[i], (*m_begin)[j], rSquared))
							continue;
						Simplex2d nSimplex = {*cm_begin, (*m_begin)[i], (*m_begin)[j]};
						std::sort(nSimplex.begin(), nSimplex.end(), comparePoints);
						mySet.insert(nSimplex);

						
					}
				}
			}
			std::cout << "adding "<<range.begin()<<" to "<<range.end()<<std::endl;
			tbb::spin_mutex::scoped_lock lock(mutex);
			    for (const auto& elem : mySet) {
				set.insert(elem);
			    }
			    lock.release();
			std::cout << "finished "<<range.begin()<<" to "<<range.end()<<std::endl;

			});
#else
	for(;cbegin != cend; ++cbegin, ++begin, ++checked){
		utils::printProgressBar(checked, all_points.size());
		/* std::cout << checked<<"/"<<all_points.size()<<std::endl; */
		for(uint32_t i = 0; i < begin->size(); ++i){
			for(uint32_t j = i+1; j < begin->size(); ++j){
				if(equal(*cbegin, (*begin)[i], (*begin)[j]))
					continue;
				if(!in_radius((*begin)[i], (*begin)[j], rSquared))
					continue;
				Simplex2d nSimplex = {*cbegin, (*begin)[i], (*begin)[j]};
				std::sort(nSimplex.begin(), nSimplex.end(), comparePoints);
				/* if (!((set.find(nSimplex[0]) != set.end()) ||  (set.find(nSimplex[1]) != set.end()) || (set.find(nSimplex[2]) != set.end()))) */
				if(set.find(nSimplex) != set.end())
					continue;
				/* if(already_simplex(nSimplex, simplexes)) */
				/* 	continue; */
				/* set.insert(nSimplex[0]); */
				/* set.insert(nSimplex[1]); */
				/* set.insert(nSimplex[2]); */
				set.insert(nSimplex);
				simplexes.emplace_back(nSimplex);

			}
		}
	}
#endif
	std::cout << "created "<<simplexes.size()<<" simplexes"<<std::endl;
	

}

inline int64_t the_lowest_x(const Simplex2d& x){
	return std::min(x[0].first, std::min(x[1].first, x[2].first));
}

inline int64_t the_lowest_y(const Simplex2d& x){
	return std::min(x[0].second, std::min(x[1].second, x[2].second));
}

inline int64_t the_highest_x(const Simplex2d& x){
	return std::max(x[0].first, std::max(x[1].first, x[2].first));
}

inline int64_t the_highest_y(const Simplex2d& x){
	return std::max(x[0].second, std::max(x[1].second, x[2].second));
}

inline void print_simplex(const Simplex2d& x){
	std::cout << "{("<<x[0].first<<','<<x[0].second<<"),("<<x[1].first<<','<<x[1].second<<"),("<<x[2].first<<','<<x[2].second<<")}"<<std::endl;;
}


simplexes_2d::simplexes_2d(const points_2d& points, int64_t radius_high, int64_t radius_low){
	std::vector<Point2d > all_points = points.generatePoints();
	const int64_t rSquared = radius_high * radius_high;
	auto cbegin = all_points.cbegin();
	auto cend = all_points.cend();
	std::cout << "have "<<all_points.size() << " to check"<<std::endl;
	/* std::cout<<"in range "<<radius_low<<',' */
	int64_t checked = 0;
	std::vector<std::vector<Point2d > > all_radi;
	all_radi.reserve(all_points.size());

	for(;cbegin != cend; ++cbegin, ++checked){
		utils::printProgressBar(checked, all_points.size());
		/* std::cout << checked<<"/"<<all_points.size()<<std::endl; */
		all_radi.push_back(points.in_radius_vec(*cbegin, radius_high, radius_low));
	}
	
	std::cout <<"got all in radi "<< std::endl;
	cbegin = all_points.cbegin();
	auto begin = all_radi.begin();
	checked = 0;
	unordered_simplex2d_set set;
	/* unordered_point2d_set set; */
	
#ifdef USE_PARALLEL
	tbb::spin_mutex mutex;
	tbb::blocked_range<size_t> b(0, all_points.size());
	std::cout<<"b grain size:"<<b.grainsize()<<", size: "<<b.size()<<std::endl;
	std::atomic_int64_t check;
	check.store(static_cast<int64_t>(0));
	tbb::parallel_for(b,
			[&](const tbb::blocked_range<size_t>& range) {
			check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
			utils::printThreadingProgressBar(check.load(), all_points.size());
			/* std::cout << "doing "<<range.begin()<<" to "<<range.end()<<" "<<range.grainsize()<<std::endl; */
			auto m_begin = all_radi.cbegin() + range.begin();
			auto m_end = all_radi.cbegin() + range.end();
			auto cm_begin = all_points.cbegin() + range.begin();
			unordered_simplex2d_set mySet;
			std::unordered_map<Point2d, unordered_simplex2d_set, Point2dHash> myMap;
			for(;m_begin != m_end; ++m_begin, ++cm_begin){
				for(uint32_t i = 0; i < m_begin->size(); ++i){
					for(uint32_t j = i+1; j < m_begin->size(); ++j){
						if(equal(*cm_begin, (*m_begin)[i], (*m_begin)[j]))
							continue;
						if(!in_radius((*m_begin)[i], (*m_begin)[j], rSquared))
							continue;
						Simplex2d nSimplex = {*cm_begin, (*m_begin)[i], (*m_begin)[j]};
						std::sort(nSimplex.begin(), nSimplex.end(), comparePoints);
						mySet.insert(nSimplex);
						myMap[nSimplex[0]].insert(nSimplex);
						
					}
				}
			}
			/* std::cout << "adding "<<range.begin()<<" to "<<range.end()<<std::endl; */
			
			tbb::spin_mutex::scoped_lock lock(mutex);
			set.insert(mySet.begin(), mySet.end());
			for(const auto& pair : myMap)
				this->simplex_map.insert(pair);
			lock.release();

			/* std::cout << "finished "<<range.begin()<<" to "<<range.end()<<std::endl; */

			});
#else


	for(;cbegin != cend; ++cbegin, ++begin, ++checked){
		utils::printProgressBar(checked, all_points.size());
		/* std::cout << checked<<"/"<<all_points.size()<<std::endl; */
		/* if(begin->size() < 2){ */
		/* 	std::cout << "some of the sizes were less than 2"<<std::endl; */
		/* } */
		for(uint32_t i = 0; i < begin->size(); ++i){
			for(uint32_t j = i+1; j < begin->size(); ++j){
				Simplex2d nSimplex = {*cbegin, (*begin)[i], (*begin)[j]};
				/* if(set.find(nSimplex) != set.end()) */
				/* 	continue; */
				if(equal(*cbegin, (*begin)[i], (*begin)[j]))
					continue;
				if(!in_radius((*begin)[i], (*begin)[j], rSquared))
					continue;

				/* if (!((set.find(nSimplex[0]) != set.end()) ||  (set.find(nSimplex[1]) != set.end()) || (set.find(nSimplex[2]) != set.end()))) */

				/* if(already_simplex(nSimplex, simplexes)) */
				/* 	continue; */
				/* set.insert(nSimplex[0]); */
				/* set.insert(nSimplex[1]); */
				/* set.insert(nSimplex[2]); */
				set.insert(nSimplex);
				this->simplex_map[nSimplex[0]].insert(nSimplex);
				/* simplexes.emplace_back(std::move(nSimplex)); */

			}
		}
	}
#endif
	std::cout << "created "<<set.size()<<" simplexes"<<std::endl;
	simplexes = std::vector<Simplex2d>(set.begin(), set.end());
}


inline uint8_t isInside(const Simplex2d& inner, const Simplex2d& outter){
	if((inner[0].first <= outter[0].first
			&& inner[0].second <= outter[0].second)
		&& (inner[1].first <= outter[1].first
			&& inner[1].second <= outter[1].second)
		&& (inner[2].first <= outter[2].first
			&& inner[2].second <= outter[2].second))
		return 1;
	if((inner[0].first >= outter[0].first
			&& inner[0].second >= outter[0].second)
		&& (inner[1].first >= outter[1].first
			&& inner[1].second >= outter[1].second)
		&& (inner[2].first >= outter[2].first
			&& inner[2].second >= outter[2].second))
		return 2;
	return 0;

}


inline void addSimplexFilterVector(std::vector<Simplex2d>& simps, const Simplex2d& simp){
	for(uint32_t i = 0; i < simps.size(); ++i){
		uint8_t choice = isInside(simps[i], simp);
		if(choice == 1){
			simps[i] = simp;
			return;
		}
		if(choice == 2){
			return;
		}
	}
	simps.push_back(simp);
}

bool mapIsInside(const Simplex2d& inner, const Simplex2d& outter){
	return (inner[1].first < outter[1].first && inner[1].second < outter[1].second) 
		&& (inner[2].first < outter[2].first
			&& inner[2].second < outter[2].second); 
}


template< class Key, class Hash, class KeyEqual, class Alloc,
          class Pred >
typename std::unordered_set<Key, Hash, KeyEqual, Alloc>::size_type
    erase_if( std::unordered_set<Key, Hash, KeyEqual, Alloc>& c,
              Pred pred ){
	auto old_size = c.size();
	for (auto first = c.begin(), last = c.end(); first != last;)
	{
	    if (pred(*first))
		first = c.erase(first);
	    else
		++first;
	}
	return old_size - c.size();	
}

simplexes_2d::simplexes_2d(const points_2d& points, int64_t radius_high, int64_t radius_low, detail::FilterSimplexes){
	std::vector<Point2d > all_points = points.generatePoints();
	const int64_t rSquared = radius_high * radius_high;
	auto cbegin = all_points.cbegin();
	auto cend = all_points.cend();
	std::cout << "have "<<all_points.size() << " to check"<<std::endl;
	/* std::cout<<"in range "<<radius_low<<',' */
	int64_t checked = 0;
	std::vector<std::vector<Point2d > > all_radi;
	all_radi.reserve(all_points.size());

	for(;cbegin != cend; ++cbegin, ++checked){
		utils::printProgressBar(checked, all_points.size());
		/* std::cout << checked<<"/"<<all_points.size()<<std::endl; */
		all_radi.push_back(points.in_radius_filtered(*cbegin, radius_high, radius_low));
	}


	
	std::cout <<std::endl<<"got all in radi "<< std::endl;
	cbegin = all_points.cbegin();
	/* auto begin = all_radi.begin(); */
	checked = 0;
	unordered_simplex2d_set set;
	/* unordered_point2d_set set; */
	auto filterFunction = [&](unordered_simplex2d_set& set){
		unordered_simplex2d_set cpySet = set;
		Simplex2d cpy = *set.begin();
		auto count = erase_if(set, [&cpy](const Simplex2d& simp){return mapIsInside(simp, cpy);});
		while(count > 0){
			cpy = *set.begin();
			count = erase_if(set, [&cpy](const Simplex2d& simp){return mapIsInside(simp, cpy);});
		}
		

		/* for(auto it = set.begin(); it != set.end();){ */

		/* 	bool it2Erased = false; */
		/* 	for(auto it2 = set.begin(); it2 != set.end(); ++it2){ */
		/* 		if(mapIsInside(*it2, *it)){ // it2 is inside it */
		/* 			it2 = set.erase(it2); */
		/* 			it2Erased = true; */
		/* 			it = set.begin(); */
		/* 			break; */
		/* 		} */
		/* 		else if(mapIsInside(*it, *it2)){ */
		/* 			it = set.erase(it); */
		/* 			++it2; */
		/* 		} */
		/* 		else{ */
		/* 			++it2; */
		/* 		} */
		/* 	} */
		/* 	if(it2Erased){continue;} */
		/* 	++it; */
		/* } */
	};

#ifdef USE_PARALLEL
	tbb::spin_mutex mutex;
	tbb::blocked_range<size_t> b(0, all_points.size());
	std::atomic_int64_t check;
	check.store(static_cast<int64_t>(0));
	std::atomic_uint64_t simplex_amt;
	simplex_amt.store(static_cast<uint64_t>(0));
	std::cout << "creating simplexes now..."<<std::endl;
	tbb::parallel_for(b,
			[&](const tbb::blocked_range<size_t>& range) {
			check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
			utils::printThreadingProgressBar(check.load(), all_points.size());
			/* std::cout << "doing "<<range.begin()<<" to "<<range.end()<<" "<<range.grainsize()<<std::endl; */
			auto m_begin = all_radi.cbegin() + range.begin();
			auto m_end = all_radi.cbegin() + range.end();
			auto cm_begin = all_points.cbegin() + range.begin();
			unordered_simplex2d_set mySet;
			std::unordered_map<Point2d, unordered_simplex2d_set, Point2dHash> myMap;
			for(;m_begin != m_end; ++m_begin, ++cm_begin){
				std::vector<Simplex2d> cur_simps;
				cur_simps.reserve(m_begin->size());
				for(uint32_t i = 0; i < m_begin->size(); ++i){
					for(uint32_t j = i+1; j < m_begin->size(); ++j){
						if(equal(*cm_begin, (*m_begin)[i], (*m_begin)[j]))
							continue;
						if(!in_radius((*m_begin)[i], (*m_begin)[j], rSquared))
							continue;
						Simplex2d nSimplex = {*cm_begin, (*m_begin)[i], (*m_begin)[j]};
						std::sort(nSimplex.begin(), nSimplex.end(), comparePoints);
						addSimplexFilterVector(cur_simps, nSimplex);
						/* myMap[nSimplex[0]].insert(nSimplex); */
						/* cur_simps.push_back(nSimplex); //checking if it is the add simplex one? */
						
					}
				}
				simplex_amt.fetch_add(cur_simps.size(), std::memory_order_relaxed);
				mySet.insert(cur_simps.begin(), cur_simps.end());
			}
			/* std::cout << "adding "<<range.begin()<<" to "<<range.end()<<std::endl; */
			
			tbb::spin_mutex::scoped_lock lock(mutex);
			for(const auto& simp : mySet){
				set.insert(simp);
				this->simplex_map[simp[0]].insert(simp);
			}
			lock.release();

			/* std::cout << "finished "<<range.begin()<<" to "<<range.end()<<std::endl; */

			});

	std::cout << std::endl<<"first filter got "<<  simplex_amt.load() << "now going to filter map..."<<std::endl;
	simplex_amt.store(static_cast<uint64_t>(0));
	check.store(static_cast<int64_t>(0));
	
	auto begin = this->simplex_map.begin();
	tbb::parallel_for(tbb::blocked_range<size_t>(0, this->simplex_map.size()),
			[&](const auto& range) {
			check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
			utils::printThreadingProgressBar(check.load(), this->simplex_map.size());
			auto it = begin;
			std::advance(it, range.begin());
			auto end = begin;
			std::advance(end, range.end());
			for(; it != end; ++it) {
				filterFunction(it->second);
				simplex_amt.fetch_add(it->second.size(), std::memory_order_relaxed);
			}

			});
	this->simplexes.reserve(simplex_amt.load());
	for(auto& pair : this->simplex_map)
		this->simplexes.insert(this->simplexes.end(), pair.second.begin(), pair.second.end());
	std::cout << "created "<<this->simplexes.size()<<" simplexes "<<simplex_amt.load()<<std::endl;

#else


	for(;cbegin != cend; ++cbegin, ++begin, ++checked){
		utils::printProgressBar(checked, all_points.size());
		/* std::cout << checked<<"/"<<all_points.size()<<std::endl; */
		/* if(begin->size() < 2){ */
		/* 	std::cout << "some of the sizes were less than 2"<<std::endl; */
		/* } */
		for(uint32_t i = 0; i < begin->size(); ++i){
			for(uint32_t j = i+1; j < begin->size(); ++j){
				Simplex2d nSimplex = {*cbegin, (*begin)[i], (*begin)[j]};
				/* if(set.find(nSimplex) != set.end()) */
				/* 	continue; */
				if(equal(*cbegin, (*begin)[i], (*begin)[j]))
					continue;
				if(!in_radius((*begin)[i], (*begin)[j], rSquared))
					continue;

				/* if (!((set.find(nSimplex[0]) != set.end()) ||  (set.find(nSimplex[1]) != set.end()) || (set.find(nSimplex[2]) != set.end()))) */

				/* if(already_simplex(nSimplex, simplexes)) */
				/* 	continue; */
				/* set.insert(nSimplex[0]); */
				/* set.insert(nSimplex[1]); */
				/* set.insert(nSimplex[2]); */
				set.insert(nSimplex);
				this->simplex_map[nSimplex[0]].insert(nSimplex);
				/* simplexes.emplace_back(std::move(nSimplex)); */

			}
		}
	}
	checked = 0;
	uint64_t simplex_amt = 0;
	for(auto begin = this->simplex_map.begin(); begin != this->simplex_map.end(); ++begin, ++checked){
		utils::printProgressBar(checked, this->simplex_map.size());
		filterFunction(begin->second);
		simplex_amt += begin->second.sixe();
	}
	this->simplexes.reserve(simplex_amt);
	for(auto& pair : this->simplex_map)
		this->simplexes.insert(this->simplexes.end(), pair.second.begin(), pair.second.end());
	std::cout << "created "<<this->simplexes.size()<<" simplexes "<<simplex_amt<<std::endl;
	

#endif
}

}
}
