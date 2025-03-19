#include "../../Tensor.h"
#include "../../utils/utils.h"
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

void addSimplex(const std::vector<Basis>& connected, const Point& point, unordered_simplex_set& simplexes, std::unordered_map<Point, unordered_simplex_set, PointHash >& simplex_map,
                int64_t on_dim){
	if(!allBallsOverlap(connected)){return;}
	size_t N = point.size();
	Simplex cur_simplex = Simplex(on_dim, N);
	cur_simplex[0].share(point);
	for(uint32_t i = 0; i < on_dim-1; ++i){
		cur_simplex[i+1].share(connected[i].center);
	}
	std::sort(cur_simplex.begin(), cur_simplex.end(), comparePoints);
	if(simplexes.find(cur_simplex) != simplexes.end())
		return;
	simplexes.insert(cur_simplex);
	for(uint32_t i = 0; i < on_dim; ++i){
		simplex_map[cur_simplex[i]].insert(cur_simplex);
	}
}

Simplexes::Simplexes(const BasisOverlapping& balls, int64_t on_dim)
	:dim(balls.dims()), simplex_dim(0)
{
    // std::cout << "input on dim is "<<on_dim<<std::endl;
    const int64_t& N = balls.dims();
    on_dim = (on_dim == -1) ? N+1 : on_dim + 1;
    simplex_dim = on_dim;
    if(simplex_dim == 1){ //then this is just individual points
        this->simplexes.reserve(balls.points.size());
        for(const auto& point : balls.points){
            Simplex cur_simplex(1, point.size());
            cur_simplex[0] = point;
            this->simplexes.insert(cur_simplex);
            this->simplex_map[cur_simplex[0]].insert(cur_simplex);
            
        }
        return;
    }
    for(const auto& point : balls.points){
        std::vector<Basis> connected = balls.getConnected(point);
        if(connected.size() < on_dim)
            continue;
        // std::cout << "connected size is "<<connected.size()<<std::endl;
        // std::cout << "simplex dim is "<<simplex_dim<<std::endl;
        //now have to decide if they all overlap with each other
        if(connected.size() == on_dim-1){
            addSimplex(connected, point, this->simplexes, this->simplex_map, on_dim);
        }
        std::vector<size_t> indices(on_dim-1);
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<Basis> combination;
        do{
            if (next_combination(connected, combination, indices, on_dim-1)) {

                addSimplex(combination, point, this->simplexes, this->simplex_map, on_dim);
            }
            else{
                break;
            }
        } while(true);
    }
}

}
}
