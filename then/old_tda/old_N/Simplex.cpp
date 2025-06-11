#include "../Tensor.h"
#include "../utils/utils.h"
#include "Simplex.h"
#include "Basis.h"
#include "Points.h"
#include <algorithm>

namespace nt{
namespace tda{


template<size_t N>
bool comparePoints(const Point<N>& p1, const Point<N>& p2){
	return sum<N>(p1) > sum<N>(p2);
}

template<typename T>
bool next_combination(const std::vector<T>& elements, std::vector<T>& combination, std::vector<size_t>& indices, size_t N) {
    size_t n = elements.size();
    combination.clear();
    combination.resize(N);
    size_t last_index = N - 1; // Index of the last element in the combination
    
    do {
        // Populate the combination using the selected indices
        for (size_t i = 0; i < N; ++i) {
            combination[i] = elements[indices[i]];
        }

        // Move to the next combination by updating the indices
        size_t i = N;
        while (i-- > 0 && indices[i] == n - N + i);
        if (indices[0] == n - N) break; // All combinations generated

        ++indices[i];
        while (++i < N) {
            indices[i] = indices[i - 1] + 1;
        }

        return true; // Combination found
    } while (true);
    
    return false; // No more combinations
}

template<size_t N>
bool allBallsOverlap(const std::vector<Basis<N>>& connected){
	for(uint32_t i = 0; i < connected.size(); ++i){
		for(uint32_t j = i+1; j < connected.size(); ++j){
			if(!connected[i].intersect(connected[j]))
				return false;
		}
	}
	return true;
}

template<size_t N>
void addSimplex(const std::vector<Basis<N>>& connected, const Point<N>& point, unordered_simplex_set<N>& simplexes, std::unordered_map<Point<N>, unordered_simplex_set<N>, PointNHash<N> >& simplex_map){
	if(!allBallsOverlap(connected)){return;}
	Simplex<N> cur_simplex;
	cur_simplex[0] = point;
	for(uint32_t i = 0; i < N; ++i){
		cur_simplex[i+1] = connected[i].center;
	}
	std::sort(cur_simplex.begin(), cur_simplex.end(), comparePoints<N>);
	if(simplexes.find(cur_simplex) != simplexes.end())
		return;
	simplexes.insert(cur_simplex);
	for(uint32_t i = 0; i < N+1; ++i){
		simplex_map[cur_simplex[i]].insert(cur_simplex);
	}
}

template<size_t N>
Simplexes<N>::Simplexes(const BasisOverlapping<N>& balls){
    for(const auto& point : balls.points){
         std::vector<Basis<N> > connected = balls.getConnected(point);
         if(connected.size() < N)
             continue;
         //now have to decide if they all overlap with each other
        if(connected.size() == N){
            addSimplex(connected, point, this->simplexes, this->simplex_map);
        }
	std::vector<size_t> indices(N);
	std::iota(indices.begin(), indices.end(), 0);

        std::vector<Basis<N>> combination;
        do{
            if (next_combination(connected, combination, indices, N)) {

                addSimplex(combination, point, this->simplexes, this->simplex_map);
            }
            else{
                break;
            }
        } while(true);
    }
}


template class Simplexes<1>;
template class Simplexes<2>;
template class Simplexes<3>;
template class Simplexes<4>;
template class Simplexes<5>;
template class Simplexes<6>;
template class Simplexes<7>;
template class Simplexes<8>;
template class Simplexes<9>;
template class Simplexes<10>;
template class Simplexes<11>;
template class Simplexes<12>;
template class Simplexes<13>;
template class Simplexes<14>;
template class Simplexes<15>;
template class Simplexes<16>;
template class Simplexes<17>;
template class Simplexes<18>;
template class Simplexes<19>;
template class Simplexes<20>;

}
}
