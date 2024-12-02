#ifndef TDA_SIMPLEX_H
#define TDA_SIMPLEX_H

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


class Simplex{
	intrusive_ptr<intrusive_list<Point>> ptr;
	public:
		Simplex() = delete;
		Simplex(int64_t n)
			:ptr(make_intrusive<intrusive_list<Point> >(n, Point(n-1)))
		{
			utils::throw_exception(n > 0, "Cannot create simplex of size less than 1");
		}
		Simplex& operator=(const Simplex&) = delete;
		inline Point& operator[](int64_t n) noexcept {return ptr->at(n);}
		inline const Point& operator[](int64_t n) const noexcept {return ptr->at(n);}
		inline const int64_t& size() const noexcept {return ptr->size();}
		inline const int64_t& dims() const noexcept {return ptr->size();}
		inline Point* begin() noexcept {return ptr->ptr();}
		inline Point* end() noexcept {return ptr->end();}
		inline const Point* begin() const noexcept {return ptr->ptr();}
		inline const Point* end() const noexcept {return ptr->end();}
		inline const Point* cbegin() const noexcept {return ptr->ptr();}
		inline const Point* cend() const noexcept {return ptr->end();}
		inline Simplex clone() const noexcept {
			Simplex out(size());
			auto o_begin = out.begin();
			for(auto begin = cbegin(); begin != cend(); ++begin)
				*o_begin = begin->clone();
			return std::move(out);
		}
		inline const Point& back() const noexcept {return (*this)[size()-1];}
		inline Point& back() noexcept {return (*this)[size()-1];}
		inline const bool operator==(const Simplex& p) const noexcept{
			if(p.size() != size()){return false;}
			auto p_b = p.begin();
			for(auto begin = cbegin(); begin != cend(); ++begin, ++p_b)
				if(*begin != *p_b){return false;}
			return true;
		}
		inline const bool operator!=(const Simplex& p) const noexcept{return !((*this) == p);}

};


inline Simplex GenerateSimplex(size_t N){
	return Simplex(N+1);
}

inline std::ostream& operator << (std::ostream& out, const Simplex& simp){
	out << '{';
	for(uint32_t i = 0; i < simp.size()-1; ++i)
		out << simp[i] << ',';
	out << simp.back() << '}';
	return out;
}


struct SimplexHash{
	inline std::size_t operator()(const Simplex& simplex) const{
		std::string key = "";
		for(uint32_t i = 0; i < simplex.size(); ++i){
			hash_helper(key, simplex[i]);
			key.back() = ';';
		}
		return std::hash<std::string>{}(key);
	}
	private:
		inline void hash_helper(std::string& str, const Point& p) const{
			for(size_t i = 0; i < p.size()-1; ++i){
				str += std::to_string(p[i]) + ',';
			}
			str += std::to_string(p.back());
		}
};

using unordered_simplex_set = typename std::unordered_set<Simplex, SimplexHash>;


class Simplexes{
	unordered_simplex_set simplexes;
	std::unordered_map<Point, unordered_simplex_set, PointHash> simplex_map;
	const int64_t dim;
	public:
		Simplexes() = delete;
		Simplexes(const BasisOverlapping&);
		inline bool isConnected(const Point& p) const 
		{return simplex_map.find(p) != simplex_map.end();}
		inline const unordered_simplex_set& operator[](const Point& p) const 
		{return simplex_map.at(p);}
		inline const int64_t& dims() const noexcept {return dim;}
		inline unordered_simplex_set::iterator begin() noexcept {return simplexes.begin();}
		inline unordered_simplex_set::iterator end() noexcept {return simplexes.end();}
		inline unordered_simplex_set::const_iterator begin() const noexcept {return simplexes.begin();}
		inline unordered_simplex_set::const_iterator end() const noexcept {return simplexes.end();}
		inline unordered_simplex_set::const_iterator cbegin() const noexcept {return simplexes.cbegin();}
		inline unordered_simplex_set::const_iterator cend() const noexcept {return simplexes.cend();}
		inline unordered_simplex_set::size_type size() const noexcept {return simplexes.size();}


};

}
}

#endif
