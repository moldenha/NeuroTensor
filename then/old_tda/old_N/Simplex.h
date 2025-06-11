#ifndef _NT_OLD_TDA_SIMPLEX_N_H_
#define _NT_OLD_TDA_SIMPLEX_N_H_

#include "../Tensor.h"
#include "../utils/utils.h"
#include "Basis.h"
#include <_types/_uint32_t.h>
#include <array>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include "Points.h"
#include <unordered_set>
#include <unordered_map>
#include <utility>

namespace nt{
namespace tda{


namespace detail{
struct FilterSimplexes {};
}


using Simplex2d = std::array<Point2d, 3>;

struct Simplex2dHasher {
    std::size_t operator()(const Simplex2d& arr) const {
        std::string key;
        for (const auto& pair : arr) {
            key += std::to_string(pair.first) + "," + std::to_string(pair.second) + ";";
        }
        return std::hash<std::string>{}(key);
    }
};

using unordered_simplex2d_set = typename std::unordered_set<Simplex2d, Simplex2dHasher>;

class simplexes_2d{
	std::vector<Simplex2d> simplexes; //this holds all the simplexes
					//a single simplex is an std::array<std::pair<int64_t, int64_t>, 3>
	std::unordered_map<Point2d, unordered_simplex2d_set, Point2dHash> simplex_map;
	
	public:
		simplexes_2d() = delete;
		simplexes_2d(const points_2d&, int64_t); // the points and a radius for it
		simplexes_2d(const points_2d&, int64_t, int64_t); // the points and a radius_high, radius_low for it
		simplexes_2d(const points_2d&, int64_t, int64_t, detail::FilterSimplexes); // the points and a radius_high, radius_low for it
		inline const std::vector<Simplex2d>& getSimplexes() const {return simplexes;}
		inline const std::unordered_map<Point2d, unordered_simplex2d_set, Point2dHash>& getSimplexMap() const {return simplex_map;}
	

};

template<size_t N>
using Simplex = std::array<Point<N>, N+1>;


template<size_t N>
inline std::ostream& operator << (std::ostream& out, const Simplex<N>& simp){
	out << '{';
	for(uint32_t i = 0; i < N; ++i)
		out << simp[i] << ',';
	out << simp[N] << '}';
	return out;
}


template<size_t N>
struct SimplexNHash{
	inline std::size_t operator()(const Simplex<N>& simplex) const{
		std::string key = "";
		for(uint32_t i = 0; i < N+1; ++i){
			hash_helper(key, simplex[i], std::make_index_sequence<N>());
			key.back() = ';';
		}
		return std::hash<std::string>{}(key);
	}
	private:
		template<std::size_t... Is>
		inline void hash_helper(std::string& str, const Point<N>& p, std::index_sequence<Is...>) const{
			str += ((std::to_string(std::get<Is>(p)) + ',') + ...);
		}
};

template<size_t N>
using unordered_simplex_set = typename std::unordered_set<Simplex<N>, SimplexNHash<N> >;


template<size_t N>
class Simplexes{
	unordered_simplex_set<N> simplexes;
	std::unordered_map<Point<N>, unordered_simplex_set<N>, PointNHash<N> > simplex_map;
	public:
		Simplexes() = delete;
		Simplexes(const BasisOverlapping<N>&);
		inline bool isConnected(const Point<N>& p) const 
		{return simplex_map.find(p) != simplex_map.end();}
		inline const unordered_simplex_set<N>& operator[](const Point<N>& p) const 
		{return simplex_map.at(p);}

		

};

}
}

#endif
