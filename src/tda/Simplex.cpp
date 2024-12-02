#include "../Tensor.h"
#include "../utils/utils.h"
#include "Simplex.h"
#include "Basis.h"
#include "Points.h"
#include <algorithm>

namespace nt{
namespace tda{


bool comparePoints(const Point& p1, const Point& p2){
	return sum(p1) > sum(p2);
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

bool allBallsOverlap(const std::vector<Basis>& connected){
	for(uint32_t i = 0; i < connected.size(); ++i){
		for(uint32_t j = i+1; j < connected.size(); ++j){
			if(!connected[i].intersect(connected[j]))
				return false;
		}
	}
	return true;
}

void addSimplex(const std::vector<Basis>& connected, const Point& point, unordered_simplex_set& simplexes, std::unordered_map<Point, unordered_simplex_set, PointHash >& simplex_map){
	if(!allBallsOverlap(connected)){return;}
	size_t N = point.size();
	Simplex cur_simplex = GenerateSimplex(N);
	cur_simplex[0].share(point);
	for(uint32_t i = 0; i < N; ++i){
		cur_simplex[i+1].share(connected[i].center);
	}
	std::sort(cur_simplex.begin(), cur_simplex.end(), comparePoints);
	if(simplexes.find(cur_simplex) != simplexes.end())
		return;
	simplexes.insert(cur_simplex);
	for(uint32_t i = 0; i < N+1; ++i){
		simplex_map[cur_simplex[i]].insert(cur_simplex);
	}
}

Simplexes::Simplexes(const BasisOverlapping& balls)
	:dim(balls.dims())
{
    const int64_t& N = balls.dims();
    for(const auto& point : balls.points){
         std::vector<Basis> connected = balls.getConnected(point);
         if(connected.size() < N)
             continue;
         //now have to decide if they all overlap with each other
        if(connected.size() == N){
            addSimplex(connected, point, this->simplexes, this->simplex_map);
        }
	std::vector<size_t> indices(N);
	std::iota(indices.begin(), indices.end(), 0);

        std::vector<Basis> combination;
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

}
}
