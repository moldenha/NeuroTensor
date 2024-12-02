#include "BatchKDTree.h"
#include <vector>
#include <unordered_set>
#include "Points.h"
#include <functional>

namespace nt{
namespace tda{
BatchKDTree::BatchKDTree(const BatchPoints& pts)
	:dim(pts.dims()), roots(pts.batches(), intrusive_ptr<KDNode>(nullptr))
{
	size_t max = roots.size();
	for(int64_t i = 0; i < max; ++i){
		std::vector<Point>& points = const_cast<std::vector<Point>&>(pts.generatePoints(i));
		utils::printProgressBar(i, max, " making "+std::to_string(points.size())+" trees");
		roots[i] = build(points.begin(), points.end(), 0);
	}
	utils::printProgressBar(max, max);
}

BatchKDTree::BatchKDTree(const int64_t dims, const int64_t batches)
	:dim(dims), roots(batches, intrusive_ptr<KDNode>(nullptr))
{}


void BatchKDTree::build_batch(std::vector<Point>& points, size_t num){
	roots[num] = build(points.begin(), points.end(), 0);
}


std::vector<Point> BatchKDTree::rangeSearch(const int64_t batch, const Point& P, double R){
	std::vector<Point> results;
	this->rangeSearch(roots[batch], P, R, 0, results);
	return std::move(results);
}

std::vector<Point> BatchKDTree::rangeSearch(const int64_t batch, const int64_t& x, const int64_t& y, double R){
	Point P({x, y});
	std::vector<Point> results;
	this->rangeSearch(roots[batch], P, R, 0, results);
	return std::move(results);
}

void BatchKDTree::rangeSearch(const int64_t batch, const Point& P, double R, std::vector<std::reference_wrapper<const Point>>& pts){
	this->rangeSearch(roots[batch], P, R, 0, pts);
}
void BatchKDTree::rangeSearch(const int64_t batch, const int64_t& x, const int64_t& y, double R, std::vector<std::reference_wrapper<const Point>>& pts){
	Point P({x, y});
	this->rangeSearch(roots[batch], P, R, 0, pts);
}


inline double distance_squared(const Point& P, const intrusive_ptr<KDNode>& node, const int64_t& dim) noexcept{
	if(dim == 2){
		return (P[0] - node->point[0]) * (P[0] - node->point[0]) + (P[1] - node->point[1]) * (P[1] - node->point[1]);
	}
	double distance = 0;
	for(int i = 0; i < dim; ++i){
		distance += (P[i] - node->point[0]) * (P[i] - node->point[0]);
	}
	return distance;
}

intrusive_ptr<KDNode> BatchKDTree::build(std::vector<Point>::iterator start,
					std::vector<Point>::iterator end,
					int depth){
	// Base case: empty range
	if (start == end) return intrusive_ptr<KDNode>(nullptr);
	
	// Alternate between x and y dimensions (0 for x, 1 for y)
	int64_t axis = depth % this->dim;

	// Find median using nth_element, which is faster than sort
	auto medianIter = start + std::distance(start, end) / 2;
	auto comparator = [&axis](const nt::tda::Point& p1, const nt::tda::Point& p2) {
		return p1[axis] < p2[axis];
	};

	std::nth_element(start, medianIter, end, comparator);
	
	// Create node and recursively build subtrees
	intrusive_ptr<KDNode> node = make_intrusive<KDNode>(*medianIter);
	
	// Recursively build the left and right subtrees
	node->left = build(start, medianIter, depth + 1);
	node->right = build(medianIter + 1, end, depth + 1);

	return node;
}


void BatchKDTree::rangeSearch(const intrusive_ptr<KDNode>& node, const Point& P, double R, int depth, std::vector<Point>& results) {
	if (!node) return;

	// Check if the point is within the radius (assumes dim = 2) (for now)
	if(this->dim == 2){
		double distanceSquared = (P[0] - node->point[0]) * (P[0] - node->point[0]) + (P[1] - node->point[1]) * (P[1] - node->point[1]);
		if (distanceSquared <= R * R) {
		    results.push_back(node->point);
		}
	}else{
		double distanceSquared = distance_squared(P, node, this->dim);
		if (distanceSquared <= std::pow(R, this->dim)) {
		    results.push_back(node->point);
		}
	}

	// Determine which side(s) to search
	int axis = depth % this->dim;
	const int64_t& pointCoordinate = P[axis];
	const int64_t& nodeCoordinate = node->point[axis];

	// Search left subtree
	if (pointCoordinate - R <= nodeCoordinate) {
	    rangeSearch(node->left, P, R, depth + 1, results);
	}

	// Search right subtree
	if (pointCoordinate + R >= nodeCoordinate) {
	    rangeSearch(node->right, P, R, depth + 1, results);
	}
}

void BatchKDTree::rangeSearch(const intrusive_ptr<KDNode>& node, const Point& P, double R, int depth, std::vector<std::reference_wrapper<const Point>>& results) {
	if (!node) return;

	// Check if the point is within the radius (assumes dim = 2) (for now)
	if(this->dim == 2){
		double distanceSquared = (P[0] - node->point[0]) * (P[0] - node->point[0]) + (P[1] - node->point[1]) * (P[1] - node->point[1]);
		if (distanceSquared <= R * R) {
		    results.push_back(std::cref(node->point));
		}
	}else{
		double distanceSquared = distance_squared(P, node, this->dim);
		if (distanceSquared <= std::pow(R, this->dim)) {
		    results.push_back(std::cref(node->point));
		}
	}

	// Determine which side(s) to search
	int axis = depth % this->dim;
	const int64_t& pointCoordinate = P[axis];
	const int64_t& nodeCoordinate = node->point[axis];

	// Search left subtree
	if (pointCoordinate - R <= nodeCoordinate) {
	    rangeSearch(node->left, P, R, depth + 1, results);
	}

	// Search right subtree
	if (pointCoordinate + R >= nodeCoordinate) {
	    rangeSearch(node->right, P, R, depth + 1, results);
	}
}

}}
