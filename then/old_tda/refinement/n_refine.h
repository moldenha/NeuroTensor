#ifndef TDA_N_REFINE_H
#define TDA_N_REFINE_H


#include "../Points.h"
#include "../Simplex.h"
#include "../../images/image.h"
#include <vector>
#include <unordered_set>
#include <utility>
#include <string>

namespace nt{
namespace tda{
namespace refine{

//next idea for optimizatons:
//filter at the points_2d class level
//this will allow ones that are below a certain radi, only get the max at a certain radi range
//this will also help as a big filter
//
//
//fixed all the issues with the points filter
//made a huge speed up with the filterSimplexes function, now takes less than a second to filter 2739229 simplexes down to 2070144
//however, the generate and merge shapes function have proven to be extremely slow, which now need to be fixed
//this ended up making the generate shapes and merge shapes the slowest functions by far though:
// going to have to redesign both the merge and generate shapes
//  current thinking:
//	- start by checking if simplex has a point within the shape
//	- if it does, automatically just add it to the shape, and then generate the points within the simplex
//	- (find a way to see if all the points were already added to the shape without generating all of them)?
//	- if the simplex does not share a point with the shape, generate the shape, then make a new
//
//
//also, for refine, start with an addition of 2 before it goes into using multiple processes
//this will cut down on getting the smallest_radius <= 1 points
//



std::vector<Tensor> generateRefinements(const Tensor&, std::vector<double>, uint8_t point=1, bool is_2d = true);


}
}
}

#endif
