#include "KDTree.h"
#include "Points.h"
#include <vector>
#include <unordered_set>

namespace nt{
namespace tda{

KDTree::KDTree(std::vector<Point>&& points)
	:dim(points.empty() ? 0 : points[0].dims()), root(build(std::move(points), 0))
{}

KDTree::KDTree(const std::vector<Point>& points)
	:dim(points.empty() ? 0 : points[0].dims()), root(build(points, 0))
{}

KDTree::KDTree(const unordered_point_set& points)
	:KDTree(std::vector<Point>(points.begin(), points.end()))
{}

std::vector<Point> KDTree::rangeSearch(const Point& P, double R){
	std::vector<Point> results;
	this->rangeSearch(root, P, R, 0, results);
	return std::move(results);
}

std::vector<Point> KDTree::rangeSearch(const int64_t& x, const int64_t& y, double R){
	Point P({x, y});
	std::vector<Point> results;
	this->rangeSearch(root, P, R, 0, results);
	return std::move(results);
}



intrusive_ptr<KDNode> KDTree::build(std::vector<Point> points, int depth){
	if (points.empty()) return intrusive_ptr<KDNode>(nullptr);
	
	//alternate between x and y dimensions (0 for x, 1 for y)
	int64_t axis = depth % this->dim;
	auto comparator = [&axis](const nt::tda::Point& p1, const nt::tda::Point& p2) {
		return p1[axis] < p2[axis];
	};

	//sort points along the selected axis
	std::sort(points.begin(), points.end(), comparator);

	//select median
	size_t medianIndex = points.size() / 2;
	const Point& medianPoint = points[medianIndex];

	//create node and recursively build subtrees
	auto node = make_intrusive<KDNode>(medianPoint);
	std::vector<Point> leftPoints(points.begin(), points.begin() + medianIndex);
	std::vector<Point> rightPoints(points.begin() + medianIndex + 1, points.end());

	node->left = build(std::move(leftPoints), depth + 1);
	node->right = build(std::move(rightPoints), depth + 1);

	return node;
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


void KDTree::rangeSearch(const intrusive_ptr<KDNode>& node, const Point& P, double R, int depth, std::vector<Point>& results) {
	if (!node) return;

	//check if the point is within the radius (assumes dim = 2) (for now)
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

	//determine which side(s) to search
	int axis = depth % this->dim;
	const int64_t& pointCoordinate = P[axis];
	const int64_t& nodeCoordinate = node->point[axis];

	//search left subtree
	if (pointCoordinate - R <= nodeCoordinate) {
	    rangeSearch(node->left, P, R, depth + 1, results);
	}

	//search right subtree
	if (pointCoordinate + R >= nodeCoordinate) {
	    rangeSearch(node->right, P, R, depth + 1, results);
	}
}


}} //nt::tda::
