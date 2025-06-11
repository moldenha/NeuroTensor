#ifndef _NT_OLD_TDA_BATCH_KD_TREE_H_
#define _NT_OLD_TDA_BATCH_KD_TREE_H_
#include <vector>
#include <unordered_set>
#include "Points.h"
#include "../../intrusive_ptr/intrusive_ptr.hpp"
#include "KDTree.h"
#include "BatchPoints.h"
#include <functional>


namespace nt{
namespace tda{

class BatchKDTree{
	int64_t dim;
	std::vector<intrusive_ptr<KDNode> > roots;
	public:
		BatchKDTree(const BatchPoints& pts);
		BatchKDTree(const int64_t dims, const int64_t batches);
		void build_batch(std::vector<Point>&, size_t);
		std::vector<Point> rangeSearch(const int64_t batch, const Point& P, double R);
		std::vector<Point> rangeSearch(const int64_t batch, const int64_t& x, const int64_t& y, double R);
		void rangeSearch(const int64_t batch, const Point& P, double R, std::vector<std::reference_wrapper<const Point>>& pts);
		void rangeSearch(const int64_t batch, const int64_t& x, const int64_t& y, double R, std::vector<std::reference_wrapper<const Point>>& pts);
		inline const int64_t& dims() const noexcept {return dim;}
		inline bool empty(int64_t batch) const noexcept {return !bool(roots[batch]);}


	private:	

		//build K-D Tree recursively
		intrusive_ptr<KDNode> build(std::vector<Point>::iterator, std::vector<Point>::iterator, int);
		// Recursive range search function
		void rangeSearch(const intrusive_ptr<KDNode>& node, const Point& P, double R, int depth, std::vector<Point>& results);
		void rangeSearch(const intrusive_ptr<KDNode>& node, const Point& P, double R, int depth, std::vector<std::reference_wrapper<const Point>>& results);
};

}} // nt::tda::

#endif // _NT_TDA_BATCH_KDTREE_H_ 
