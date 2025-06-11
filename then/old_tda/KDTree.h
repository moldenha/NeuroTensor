#ifndef _NT_OLD_TDA_KD_TREE_H_
#define _NT_OLD_TDA_KD_TREE_H_
#include <vector>
#include <unordered_set>
#include "Points.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"

namespace nt{
namespace tda{

struct KDNode : public intrusive_ptr_target {
    Point point;
    intrusive_ptr<KDNode> left, right;
    
    KDNode(const Point& p) : point(p), left(nullptr), right(nullptr) {}
};

class KDTree{
	int64_t dim;
	intrusive_ptr<KDNode> root;
	public:
		KDTree(std::vector<Point>&& points);
		KDTree(const std::vector<Point>& points);
		KDTree(const unordered_point_set& points);
		std::vector<Point> rangeSearch(const Point& P, double R);
		std::vector<Point> rangeSearch(const int64_t& x, const int64_t& y, double R);
		inline const int64_t& dims() const noexcept {return dim;}
		inline bool empty() const noexcept {return !bool(root);}


	private:	

		//build K-D Tree recursively
		intrusive_ptr<KDNode> build(std::vector<Point> points, int depth);
		//recursive range search function
		void rangeSearch(const intrusive_ptr<KDNode>& node, const Point& P, double R, int depth, std::vector<Point>& results);
};


}} //nt::tda::
#endif //_NT_TDA_KD_TREE_H_
