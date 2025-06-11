#include "../Tensor.h"
#include "../utils/utils.h"
#include <_types/_uint32_t.h>
#include <array>
#include <sys/_types/_int64_t.h>
#include <sys/types.h>
#include "Points.h"
#include "Shapes.h"
#include <cmath>
#ifdef USE_PARALLEL
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#endif
#include <atomic>
#include <random>
#include <iomanip>
#include <sstream>

#include "gnuplot_wrapper.hpp"

namespace nt{
namespace tda{


/* struct Point2dHash { */
/*     std::size_t operator()(const Point2d& p) const { */
/*         return std::hash<int64_t>{}(p.first) ^ std::hash<int64_t>{}(p.second); */
/*     } */
/* }; */

/* struct PairEqual { */
/*     bool operator()(const Point2d& p1, const Point2d& p2) const { */
/*         return p1.first == p2.first && p1.second == p2.second; */
/*     } */
/* }; */

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

inline double triangleArea(const Simplex2d& points) {
    // Extract coordinates of vertices
    int64_t x1 = points[0].first, y1 = points[0].second;
    int64_t x2 = points[1].first, y2 = points[1].second;
    int64_t x3 = points[2].first, y3 = points[2].second;

    // Calculate area using Shoelace formula
    return std::abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0);
}

inline bool equal(const Point2d& a, const Point2d& b){
	return a.first == b.first && a.second == b.second;
}

inline bool simplexes_2d_share_point(const Simplex2d &a, const Simplex2d &b){
	return a[0].first == b[0].first && a[0].second == b[0].second
		|| a[1].first == b[0].first && a[1].second == b[0].second
		|| a[2].first == b[0].first && a[2].second == b[0].second
		|| a[0].first == b[1].first && a[0].second == b[1].second 
		|| a[1].first == b[1].first && a[1].second == b[1].second
		|| a[2].first == b[1].first && a[2].second == b[1].second
		|| a[0].first == b[2].first && a[0].second == b[2].second 
		|| a[1].first == b[2].first && a[1].second == b[2].second
		|| a[2].first == b[2].first && a[2].second == b[2].second;
}

inline bool simplexes_2d_same(const Simplex2d &a, const Simplex2d &b){
	if(!(a[0].first == b[0].first && a[0].second == b[0].second
		|| a[1].first == b[0].first && a[1].second == b[0].second
		|| a[2].first == b[0].first && a[2].second == b[0].second))
		return false;
	if(!(a[0].first == b[1].first && a[0].second == b[1].second 
		|| a[1].first == b[1].first && a[1].second == b[1].second
		|| a[2].first == b[1].first && a[2].second == b[1].second))
		return false;
	if(!(a[0].first == b[2].first && a[0].second == b[2].second 
		|| a[1].first == b[2].first && a[1].second == b[2].second
		|| a[2].first == b[2].first && a[2].second == b[2].second))
		return false;
	return true;
}

inline bool is_valid_simplex(const Simplex2d &a){
	return !(equal(a[0], a[1]) || equal(a[1],a[2]) || equal(a[2], a[0])); //basically that none of them are equal
}

/* bool shape_2d::simplex_already_added(const Simplex2d &simplex) const{ */
/* 	if(!is_valid_simplex(simplex)){return false;} */
/* 	for(auto cbegin = shape.cbegin(); cbegin != shape.cend(); ++cbegin){ */
/* 		if(simplexes_2d_same(*cbegin, simplex)){return true;} */
/* 	} */
/* 	return false; */
/* } */

/* bool shape_2d::simplex_in_shape(const Simplex2d &simplex) const{ */
/* 	if(!is_valid_simplex(simplex)){return false;} */
/* 	for(auto cbegin = shape.cbegin(); cbegin != shape.cend(); ++cbegin){ */
/* 		if(simplexes_2d_share_point(*cbegin, simplex)){return true;} */
/* 	} */
/* 	return false; */
/* } */

double shape_2d::area() const{
	auto cbegin = shape.cbegin();
	auto cend = shape.cend();
	double cur = 0;
	for(;cbegin != cend; ++cbegin)
		cur += triangleArea(*cbegin);
	return cur;
}

/*
def fill_shape(tensor, points, value=3):
    # Create a meshgrid of coordinates for the tensor
    x, y = torch.meshgrid(torch.arange(tensor.shape[1]), torch.arange(tensor.shape[0]))
    x, y = x.float(), y.float()

    # Convert points to torch tensor
    points = torch.tensor(points)

    # Create a mask to check if the points are inside the shape
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        mask = mask ^ (((y0 <= y) & (y < y1)) | ((y1 <= y) & (y < y0))) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)

    # Fill the shape in the tensor with the specified value
    tensor[mask] = value*/


//the below functions are doing it using tensor functions, I have a different version for uint8_t tensors
//for now just going to fill a simplex 
void fill_tensor_simplex(const Simplex2d& simplex, Tensor& toF, Scalar& value){
	Tensor xy = functional::meshgrid(functional::arange(toF.shape()[0], DType::Float32), functional::arange(toF.shape()[1], DType::Float32));
	Tensor x = xy[0].item<Tensor>();
	Tensor y = xy[1].item<Tensor>();

	/* denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]) */
	float denom = static_cast<float>((simplex[1].second - simplex[2].second) * (simplex[0].first - simplex[2].first) + (simplex[2].first - simplex[1].first) * (simplex[0].second - simplex[2].second));
	//b0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom

	Tensor b0 = (static_cast<float>(simplex[1].second - simplex[2].second) * (x - static_cast<float>(simplex[2].first)) + static_cast<float>(simplex[2].first - simplex[1].first) * (y - static_cast<float>(simplex[2].second))) / denom;
	//b1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
	Tensor b1 = (static_cast<float>(simplex[2].second - simplex[0].second) * (x - static_cast<float>(simplex[2].first)) + static_cast<float>(simplex[0].first - simplex[2].first) * (y - static_cast<float>(simplex[2].second))) / denom;
	Tensor b2 = 1 - b0 - b1;
	Tensor mask = (b0 >= 0) && (b1 >= 0) && (b2 >= 0);
	toF[mask] = value;
}

void fill_tensor_simplex(const Simplex2d& simplex, Tensor& toF, Scalar& value, const Tensor& x, const Tensor& y){

	/* denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]) */
	float denom = static_cast<float>((simplex[1].second - simplex[2].second) * (simplex[0].first - simplex[2].first) + (simplex[2].first - simplex[1].first) * (simplex[0].second - simplex[2].second));
	//b0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom

	Tensor b0 = (static_cast<float>(simplex[1].second - simplex[2].second) * (x - static_cast<float>(simplex[2].first)) + static_cast<float>(simplex[2].first - simplex[1].first) * (y - static_cast<float>(simplex[2].second))) / denom;
	//b1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
	Tensor b1 = (static_cast<float>(simplex[2].second - simplex[0].second) * (x - static_cast<float>(simplex[2].first)) + static_cast<float>(simplex[0].first - simplex[2].first) * (y - static_cast<float>(simplex[2].second))) / denom;
	Tensor b2 = 1 - b0 - b1;
	Tensor mask = (b0 >= 0) && (b1 >= 0) && (b2 >= 0);
	toF[mask] = value;
}




bool hasSharedPoint(const std::vector<unordered_point2d_set>& sets) {
    for (size_t i = 0; i < sets.size(); ++i) {
        const auto& set1 = sets[i];
        for (size_t j = i + 1; j < sets.size(); ++j) {
            const auto& set2 = sets[j];
            for (const auto& point : set1) {
                if (set2.find(point) != set2.end()) {
                    // Found shared point
                    return true;
                }
            }
        }
    }
    // No shared point found
    return false;
}

bool hasSharedPoint(const unordered_point2d_set& set_a, const unordered_point2d_set& set_b) {
	for(const auto& point: set_a){
		if(set_b.find(point) != set_b.end())
			return true;
	}
	return false;
}

bool shape_2d::shapes_overlap(const shape_2d& sh) const{
	for(const auto& point: sh.pairSet){
		if(pairSet.find(point) != pairSet.end())
			return true;
	}
	return false;
}

void shape_2d::merge_shape(shape_2d&& sh){
	pairSet.reserve(sh.pairSet.size());
	for(const auto& point : sh.pairSet){
		pairSet.insert(point);
	}
	shape.reserve(sh.shape.size());
	for(uint32_t i = 0; i < sh.shape.size(); ++i){
		shape.push_back(std::move(sh.shape[i]));
	}
}


void fillPointsInSimplex(Tensor& matrix, size_t rows, size_t cols, const Simplex2d& simplex, Scalar& value) {
    utils::throw_exception(matrix.dtype == DType::uint8, "Expected tensor to have dtype of uint8 but got $",matrix.dtype);
    utils::throw_exception(matrix.dims() == 2, "Expected tensor to have at most 2 dimensions but got $", matrix.dims());
    // Helper lambda function to check if a point is inside the simplex
    auto isInsideSimplex = [&](int64_t x, int64_t y) {
        // Compute barycentric coordinates
	    /* std::cout << "calculating inside"<<std::endl; */
	double div = ((simplex[1].second - simplex[2].second) * (simplex[0].first - simplex[2].first) +
                        (simplex[2].first - simplex[1].first) * (simplex[0].second - simplex[2].second));
	if(div == 0)
		return true;
        double alpha = ((simplex[1].second - simplex[2].second) * (x - simplex[2].first) +
                        (simplex[2].first - simplex[1].first) * (y - simplex[2].second)) /
                        div;
        double beta = ((simplex[2].second - simplex[0].second) * (x - simplex[2].first) +
                       (simplex[0].first - simplex[2].first) * (y - simplex[2].second)) /
                       div;
        double gamma = 1.0 - alpha - beta;

        // Check if the point is inside the simplex
        return alpha >= 0 && beta >= 0 && gamma >= 0;
    };

    uint8_t* mat = reinterpret_cast<uint8_t*>(matrix.data_ptr());
    uint8_t val = value.to<uint8_t>();
    int64_t min_x = std::min(simplex[2].first, std::min(simplex[0].first, simplex[1].first));
    int64_t max_x = std::max(simplex[2].first, std::max(simplex[0].first, simplex[1].first));
    int64_t min_y = std::min(simplex[2].second, std::min(simplex[0].second, simplex[1].second));
    int64_t max_y = std::max(simplex[2].second, std::max(simplex[0].second, simplex[1].second));


    if(simplex[0].first == simplex[1].first && simplex[1].first == simplex[2].first){
	    min_y = std::max(static_cast<int64_t>(0), min_y);
	    max_y = std::min(static_cast<int64_t>(cols-1), max_y);
	for(int64_t y = min_y; y <= max_y; ++y){
		mat[size_t(simplex[0].first) * cols + size_t(y)] = val;
	}
	return;
    }
    if(simplex[0].second == simplex[1].second && simplex[1].second == simplex[2].second){
	    min_x = std::max(static_cast<int64_t>(0), min_x);
	    max_x = std::min(static_cast<int64_t>(rows-1), max_x);
	for(int64_t x = min_x; x <= max_x; ++x){
		mat[size_t(x) * cols + size_t(simplex[0].second)] = val;
	}
	return;
    }

    
    min_x = std::max(static_cast<int64_t>(0), min_x-1);
    min_y = std::max(static_cast<int64_t>(0), min_y-1);
    max_x = std::min(max_x+1, static_cast<int64_t>(rows));
    max_y = std::min(max_y+1, static_cast<int64_t>(cols));
    /* std::cout << "{("<<simplex[0].first<<","<<simplex[0].second<<"),("<<simplex[1].first<<','<<simplex[1].second<<"),("<<simplex[2].first<<','<<simplex[2].second<<"}"<<std::endl; */


	

    /* std::cout << "going to fill tensor with dims: {("<<min_x<<','<<max_x<<"),("<<min_y<<','<<max_y<<")}"<<std::endl; */
    /* std::cout << "val is "<<int(val)<<std::endl; */
    // Fill points inside the simplex with 3
    for (int64_t x = min_x; x < max_x; ++x) {
        for (int64_t y = min_y; y < max_y; ++y) {
            if (isInsideSimplex(x, y)) {
                mat[size_t(x) * cols + size_t(y)] = val;
            }
        }
    }
}


void shape_2d::fill_tensor(Tensor& t, Scalar val) const{
	utils::throw_exception(t.dims() == 2, "Expected tensor to have dimensions of 2 but got $", t.dims());
	/* Tensor xy = functional::meshgrid(functional::arange(t.shape()[0], DType::Float32), functional::arange(t.shape()[1], DType::Float32)); */
	/* Tensor x = xy[0].item<Tensor>(); */
	/* Tensor y = xy[1].item<Tensor>(); */
	for(auto cbegin = shape.cbegin(); cbegin != shape.cend(); ++cbegin){
		fillPointsInSimplex(t, t.shape()[-2], t.shape()[-1], *cbegin, val);
	}
}

void shape_2d::fill_tensor(Tensor& t, Scalar val, const std::vector<Simplex2d>& simps) const{
	utils::throw_exception(t.dims() == 2, "Expected tensor to have dimensions of 2 but got $", t.dims());
	/* Tensor xy = functional::meshgrid(functional::arange(t.shape()[0], DType::Float32), functional::arange(t.shape()[1], DType::Float32)); */
	/* Tensor x = xy[0].item<Tensor>(); */
	/* Tensor y = xy[1].item<Tensor>(); */
	for(auto cbegin = simps.cbegin(); cbegin != simps.cend(); ++cbegin){
		fillPointsInSimplex(t, t.shape()[-2], t.shape()[-1], *cbegin, val);
	}
}

std::unordered_map<Point2d, std::vector<Simplex2d>, Point2dHash> groupSimplexes(const std::vector<Simplex2d>& simplexes) {
    std::unordered_map<Point2d, std::vector<Simplex2d>, Point2dHash> groupedSimplexes;
    uint32_t i = 0;

    for (const auto& simplex : simplexes) {
	utils::printProgressBar(i, simplexes.size());
        // For each point in the simplex, add the simplex to the corresponding group
        for (const auto& point : simplex) {
            groupedSimplexes[point].push_back(simplex);
        }
    }

    return groupedSimplexes;
}

// Function to perform DFS to find connected simplexes
void dfs(const std::vector<std::unordered_set<uint32_t>>& adjacencyList, uint32_t current, std::vector<bool>& visited, std::unordered_set<uint32_t>& connectedGroup) {
    visited[current] = true;
    connectedGroup.insert(current);
    for (uint32_t neighbor : adjacencyList[current]) {
        if (!visited[neighbor]) {
            dfs(adjacencyList, neighbor, visited, connectedGroup);
        }
    }
}

std::vector<std::vector<size_t>> findSetsWithSharedPoints(const std::vector<unordered_point2d_set>& sets) {
    std::vector<std::vector<size_t>> sharedSetsIndexes;

    for (size_t i = 0; i < sets.size(); ++i) {
        const auto& set1 = sets[i];
        std::vector<size_t> sharedIndexes;

        for (size_t j = 0; j < sets.size(); ++j) {
            if (i == j) continue; // Skip self-comparison
            const auto& set2 = sets[j];

            for (const auto& point : set1) {
                if (set2.find(point) != set2.end()) {
                    // Found shared point, store the index of set2
                    sharedIndexes.push_back(j);
                    break; // No need to continue checking points in set2
                }
            }
        }

        // If any shared indexes found, store them
        if (!sharedIndexes.empty()) {
            sharedSetsIndexes.push_back(std::move(sharedIndexes));
        }
    }

    return sharedSetsIndexes;
}

std::vector<shape_2d> groupConnectedSimplexes(const std::vector<Simplex2d>& simplexes) {
    // Build the adjacency list
    std::unordered_map<Point2d, std::vector<uint32_t>, Point2dHash> pointToSimplexes;
    for (uint32_t i = 0; i < simplexes.size(); ++i) {
        for (const auto& point : simplexes[i]) {
            pointToSimplexes[point].push_back(i);
        }
    }

    // Create the adjacency list from the point-to-simplexes mapping
    std::vector<std::unordered_set<uint32_t>> adjacencyList(simplexes.size());
    for (const auto& entry : pointToSimplexes) {
        const auto& simplexIndices = entry.second;
        for (uint32_t i = 0; i < simplexIndices.size(); ++i) {
            for (uint32_t j = i + 1; j < simplexIndices.size(); ++j) {
                adjacencyList[simplexIndices[i]].insert(simplexIndices[j]);
                adjacencyList[simplexIndices[j]].insert(simplexIndices[i]);
            }
        }
    }

    // Perform DFS to group connected simplexes
    std::vector<std::vector<Simplex2d>> connectedGroups;
    std::vector<unordered_point2d_set> connectedPoints;
    std::vector<bool> visited(simplexes.size(), false);
    for (uint32_t i = 0; i < simplexes.size(); ++i) {
        if (!visited[i]) {
            std::unordered_set<uint32_t> connectedGroup;
            dfs(adjacencyList, i, visited, connectedGroup);
            std::vector<Simplex2d> group;
	    unordered_point2d_set points;
            for (uint32_t index : connectedGroup) {
                group.push_back(simplexes[index]);
		points.insert(simplexes[index][0]);
		points.insert(simplexes[index][1]);
		points.insert(simplexes[index][2]);
            }
            connectedGroups.push_back(std::move(group));
	    connectedPoints.push_back(std::move(points));
        }
    }

    std::vector<shape_2d> shapes;
    shapes.reserve(connectedGroups.size());
    for(uint32_t i = 0; i < connectedGroups.size(); ++i){
	/* std::cout << connectedPoints[i].size() << ':' << connectedGroups[i].size() << std::endl; */
	shapes.push_back(shape_2d(std::move(connectedGroups[i]), std::move(connectedPoints[i])));
    }
    return std::move(shapes);
}


std::vector<shape_2d> groupConnectedSimplexes(const std::vector<Simplex2d>& simplexes, uint32_t start, uint32_t end) {
    // Build the adjacency list
    std::unordered_map<Point2d, std::vector<uint32_t>, Point2dHash> pointToSimplexes;
    for (uint32_t i = start; i < end; ++i) {
        for (const auto& point : simplexes[i]) {
            pointToSimplexes[point].push_back(i);
        }
    }

    // Create the adjacency list from the point-to-simplexes mapping
    std::vector<std::unordered_set<uint32_t>> adjacencyList(simplexes.size());
    for (const auto& entry : pointToSimplexes) {
        const auto& simplexIndices = entry.second;
        for (uint32_t i = 0; i < simplexIndices.size(); ++i) {
            for (uint32_t j = i + 1; j < simplexIndices.size(); ++j) {
                adjacencyList[simplexIndices[i] - start].insert(simplexIndices[j] - start);
                adjacencyList[simplexIndices[j] - start].insert(simplexIndices[i] - start);
            }
        }
    }

    // Perform DFS to group connected simplexes
    std::vector<std::vector<Simplex2d>> connectedGroups;
    std::vector<unordered_point2d_set> connectedPoints;
    std::vector<std::unordered_set<uint32_t> > connectedIndexes;
    std::vector<bool> visited(end - start, false);
    for (uint32_t i = start; i < end; ++i) {
        if (!visited[i - start]) {
            std::unordered_set<uint32_t> connectedGroup;
            dfs(adjacencyList, i - start, visited, connectedGroup);
            std::vector<Simplex2d> group;
	    unordered_point2d_set points;
            for (uint32_t index : connectedGroup) {
                group.push_back(simplexes[index + start]);
		points.insert(simplexes[index + start][0]);
		points.insert(simplexes[index + start][1]);
		points.insert(simplexes[index + start][2]);
            }
            connectedGroups.push_back(std::move(group));
	    connectedPoints.push_back(std::move(points));
	    connectedIndexes.push_back(std::move(connectedGroup));
        }
    }


    std::vector<shape_2d> shapes;
    shapes.reserve(connectedGroups.size());
    for(uint32_t i = 0; i < connectedGroups.size(); ++i){
	/* std::cout << connectedPoints[i].size() << ':' << connectedGroups[i].size() << std::endl; */
	shapes.push_back(shape_2d(std::move(connectedGroups[i]), std::move(connectedPoints[i])));
    }
    return std::move(shapes);
}


inline int64_t the_lowest_xa(const Simplex2d& x){
	return std::min(x[0].first, std::min(x[1].first, x[2].first));
}

inline int64_t the_lowest_ya(const Simplex2d& x){
	return std::min(x[0].second, std::min(x[1].second, x[2].second));
}

inline int64_t the_highest_xa(const Simplex2d& x){
	return std::max(x[0].first, std::max(x[1].first, x[2].first));
}

inline int64_t the_highest_ya(const Simplex2d& x){
	return std::max(x[0].second, std::max(x[1].second, x[2].second));
}

inline void print_simplexa(const Simplex2d& x){
	std::cout << "{("<<x[0].first<<','<<x[0].second<<"),("<<x[1].first<<','<<x[1].second<<"),("<<x[2].first<<','<<x[2].second<<")}"<<std::endl;;
}




std::vector<shape_2d> shapes_2d::generateShapes(const simplexes_2d& si) const{
	const std::vector<Simplex2d>& simps = si.getSimplexes();
	const uint32_t max_simplexes_at_once = 20000; // 20,000
	std::cout << "parsing "<<simps.size()<<" simplexes"<<std::endl;
	if(simps.size() == 0)
		return std::vector<shape_2d>();
	if(simps.size() <= max_simplexes_at_once){
		return groupConnectedSimplexes(simps);
	}

	std::vector<shape_2d> m_shapes;
#ifdef USE_PARALLEL
	tbb::spin_mutex mutex;
	std::atomic_int64_t check;
	check.store(static_cast<int64_t>(0));
	tbb::parallel_for(tbb::blocked_range<size_t>(0, simps.size()),
	[&](const tbb::blocked_range<size_t>& range){
	check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
	utils::printProgressBar(check.load(), simps.size());
	uint32_t start = range.begin();
	uint32_t end = std::min(start + max_simplexes_at_once, static_cast<uint32_t>(range.end()));
	/* std::cout << "doing "<<range.begin()<<" to "<<range.end()<<std::endl; */
	std::vector<shape_2d> cur_shapes = groupConnectedSimplexes(simps, start, end);
	start += max_simplexes_at_once;
	end += max_simplexes_at_once;
	for(;start < range.end(); end += max_simplexes_at_once, start += max_simplexes_at_once){
		if(end > range.end()){
			end = range.end();
		}
		std::vector<shape_2d> Ncur_shapes = groupConnectedSimplexes(simps, start, end);
		for(uint32_t i = 0; i < Ncur_shapes.size(); ++i){
			bool added = false;
			for(uint32_t j = 0; j < cur_shapes.size(); ++j){
				if(cur_shapes[j].shapes_overlap(Ncur_shapes[i])){
					cur_shapes[j].merge_shape(std::move(Ncur_shapes[i]));
					added = true;
					break;
				}
			}
			if(!added){
				cur_shapes.push_back(std::move(Ncur_shapes[i]));
			}
		}

	}
	/* std::cout << "adding "<<range.begin()<<" to "<<range.end()<<std::endl; */
	/* tbb::mutex::scoped_lock lock; */
	tbb::spin_mutex::scoped_lock lock(mutex);

	for(uint32_t i = 0; i < cur_shapes.size(); ++i){
		bool found = false;
		for(uint32_t j = 0; j < m_shapes.size(); ++j){
			if(m_shapes[j].shapes_overlap(cur_shapes[i])){
				m_shapes[j].merge_shape(std::move(cur_shapes[i]));
				found = true;
				break;
			}
		}
		if(!found){m_shapes.push_back(std::move(cur_shapes[i]));}
	}
	lock.release();
	/* std::cout << "finished "<<range.begin()<<" to "<<range.end()<<std::endl; */
	
	});
	std::cout <<std::endl<< "done with parallel for loop"<<std::endl;
#else
	uint32_t start = 0;
	uint32_t end = start + max_simplexes_at_once;
	for(;start < simps.size(); end += max_simplexes_at_once, start += max_simplexes_at_once){
		if(end > simps.size()){
			end = simps.size();
		}
		utils::printProgressBar(start, simps.size());
		/* std::cout << "looking at range "<<start<<" to "<<end<<std::endl; */
		std::vector<shape_2d> cur_shapes = groupConnectedSimplexes(simps, start, end);
		for(uint32_t i = 0; i < cur_shapes.size(); ++i){
			bool added = false;
			for(uint32_t j = 0; j < m_shapes.size(); ++j){
				if(m_shapes[j].shapes_overlap(cur_shapes[i])){
					m_shapes[j].merge_shape(std::move(cur_shapes[i]));
					added = true;
					break;
				}
			}
			if(!added){m_shapes.push_back(std::move(cur_shapes[i]));}
		}
		if(end == simps.size())
			break;
	}
#endif
	std::vector<shape_2d> out_shapes;
	out_shapes.reserve(m_shapes.size());
	out_shapes.push_back(std::move(m_shapes[0]));
	for(uint32_t i = 1; i < m_shapes.size(); ++i){
		bool added= false;

		for(uint32_t j = 0; j < out_shapes.size(); ++j){
			if(out_shapes[j].shapes_overlap(m_shapes[i])){
				out_shapes[j].merge_shape(std::move(m_shapes[i]));
				added = true;
				break;
			}
		}
		if(!added){out_shapes.push_back(std::move(m_shapes[i]));}
	}

	std::cout << "got "<<out_shapes.size()<<" groups"<<std::endl;


	return std::move(out_shapes);

}

shapes_2d::shapes_2d(const simplexes_2d& simps)
	:shapes(generateShapes(simps))
{}

shapes_2d& shapes_2d::combine_self(shapes_2d&& shape){
	for(uint32_t i = 0; i < shape.shapes.size(); ++i){
		bool found = false;
		for(uint32_t j = 0; j < shapes.size(); ++j){
			if(shapes[j].shapes_overlap(shape.shapes[i])){
				shapes[j].merge_shape(std::move(shape.shapes[i]));
				found = true;
				break;
			}
		}
		if(!found){
			shapes.push_back(std::move(shape.shapes[i]));
		}
	}
	std::vector<shape_2d> nShapes;
	if(shapes.size() == 0)
		return *this;
	nShapes.reserve(1);
	nShapes.push_back(std::move(shapes[0]));
	for(uint32_t i = 1; i < shapes.size(); ++i){
		bool found = false;
		for(uint32_t j = 0; j < nShapes.size(); ++j){
			if(nShapes[j].shapes_overlap(shapes[i])){
				nShapes[j].merge_shape(std::move(shapes[i]));
				found = true;
				break;
			}
		}
		if(!found)
			nShapes.push_back(std::move(shapes[i]));
	}
	shapes = std::move(nShapes);
	return *this;
	
}

// Function to generate a random integer between min and max (inclusive)
int getRandomInt(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

std::string generateRandomColor() {
    std::ostringstream oss;
    oss << "#" << std::hex << std::setw(6) << std::setfill('0') << (getRandomInt(0, 0xFFFFFF));
    return oss.str();
}


//hd had 55 different shapes, largest was 28, there were also multiple with 23 groups
//wt had 44 different shapes, largest was 12 groups
//
//for radius 50:
//hd had 21 groups, biggest was 162 spheres overlapping
//wt had 15 groups, biggest was 40 spheres overlapping, multiple above 10

void visualize(const Points<3>& points, uint32_t height, uint32_t width, uint32_t channels, int64_t point_radius, int64_t ball_radius, bool only_biggest_shape, uint32_t shapes_above){
	GraphDrawer drawer;
	std::vector<Point<3>> pts = points.generatePoints();
	std::cout << "generated "<<pts.size()<<" points"<<std::endl;
	Shapes<3> shapes(points);
	std::cout << "setting radius of basis's"<<std::endl;
	shapes.setRadius(ball_radius, false);
	std::cout << "setting points in plot"<<std::endl;
	for(const auto& pt : pts){
		drawer.drawPoint(std::get<1>(pt), std::get<2>(pt), std::get<0>(pt));
	}
	std::cout << "drew points"<<std::endl;
	std::vector<BasisOverlapping<3>> balls = shapes.getBalls().getBalls();
	double draw_radius = static_cast<double>(ball_radius);
	std::cout << "making "<<balls.size()<<" groups of spheres with radius "<<draw_radius<<"..."<<std::endl;

	if(!only_biggest_shape){
		for(const auto& bs : balls){
			std::string color = generateRandomColor();
			if(bs.points.size() < shapes_above)
				continue;
			std::cout << "making "<<bs.points.size()<<" spheres in this group with the designated color "<<color<<std::endl;
			for(const auto& pt : bs.points){
				/* const Basis<3>& ball = bs[pt]; */
				drawer.drawSphere(std::get<1>(pt), std::get<2>(pt), std::get<0>(pt), draw_radius, color);
			}
		}
	}
	else{
		uint32_t index = 0;
		for(uint32_t i = 1; i < balls.size(); ++i){
			if(balls[i].points.size() > balls[index].points.size())
				index = i;
		}
		const auto& bs = balls[index];
		std::string color = "#ff59d6";
		std::cout << "making "<<bs.points.size()<<" spheres in this group with the designated color "<<color<<std::endl;
		for(const auto& pt : bs.points){
			drawer.drawSphere(std::get<1>(pt), std::get<2>(pt), std::get<0>(pt), draw_radius, color);
		}
	
	}

	std::cout << "outputting"<<std::endl;
	double x = static_cast<double>(width);
	double y = static_cast<double>(height);
	double z = static_cast<double>(channels);
	double x_div = x / 2;
	double y_div = y / 2;
	drawer.display(-(x - x_div), x+x_div, -(y-y_div), y+y_div, -z, z);
	drawer.inputCommands();
}


}
}
