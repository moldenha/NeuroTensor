/* #include "refine.h" */
/* #include "../Points.h" */
/* #include "../Simplex.h" */
/* #include "../Shapes.h" */
#include "../Points.h"
#include "../Simplex.h"
#include "refine.h"

#include "../../utils/utils.h"
#include "../../Tensor.h"
#include <vector>
#include <array>
#include <cstdlib>
#ifdef USE_PARALLEL
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#endif

#include <string>


namespace nt{
namespace tda{
namespace refine{

std::vector<std::array<uint8_t, 3>> generateColors(uint32_t num) {
    // Define black color
    std::array<uint8_t, 3> black = {0, 0, 0};

    // Generate additional colors
    std::vector<std::array<uint8_t, 3>> colors;
    colors.push_back(black); // Add black color
    for (int i = 1; i <= num; ++i) {
        // Generate random colors (just as an example)
        uint8_t r = rand() % 256;
        uint8_t g = rand() % 256;
        uint8_t b = rand() % 256;
        colors.push_back({r, g, b});
    }

    return colors;
}


Shape2d::Shape2d(const std::vector<Point2d>& pts){
	points.insert(pts.begin(), pts.end());
}

Shape2d::Shape2d(unordered_point2d_set s)
	:points(std::move(s))
{}

bool Shape2d::shapes_overlap(const Shape2d& sh) const {
	if(sh.points.size() < points.size()){
		for(const Point2d& pt : sh.points){
			if(points.find(pt) != points.end())
				return true;
		}
	}
	for(const Point2d& pt : points){
		if(sh.points.find(pt) != sh.points.end())
			return true;
	}
	return false;
}

void Shape2d::merge_shape(Shape2d&& sh){
	points.merge(std::move(sh.points));
}


template<typename T>
void fillPointsInShape(Tensor& matrix, Scalar& value, const unordered_point2d_set& points){
	T val = value.to<T>();
	T* mat = reinterpret_cast<T*>(matrix.data_ptr());
	uint32_t rows = matrix.shape()[-2];
	uint32_t cols = matrix.shape()[-1];
	for(uint32_t x = 0; x < rows; ++x){
		for(uint32_t y = 0; y < cols; ++y, ++mat){
			Point2d p(x,y);
			if(points.find(p) != points.end()){
				*mat = val;
			}
		}
	}
}

void Shape2d::fill_tensor(Tensor& matrix, Scalar val) const{
	utils::throw_exception(matrix.dims() == 2, "Expected tensor to have at most 2 dimensions but got $", matrix.dims());
	utils::throw_exception(matrix.dtype == DType::uint8
			|| matrix.dtype == DType::uint16
			|| matrix.dtype == DType::uint32
			|| matrix.dtype == DType::int64
			,"Expected matrix to have unsigned integer or int64 dtype but got $", matrix.dtype);
	utils::throw_exception(matrix.is_contiguous(), "Expected matrix to be contiguous");
	switch(matrix.dtype){
		case DType::uint8:
			fillPointsInShape<uint8_t>(matrix, val, points);
			break;
		case DType::uint16:
			fillPointsInShape<uint16_t>(matrix, val, points);
			break;
		case DType::uint32:
			fillPointsInShape<uint32_t>(matrix, val, points);
			break;
		case DType::int64:
			fillPointsInShape<int64_t>(matrix, val, points);
			break;

		default:
			break;
	}
}


uint64_t mergeAllShapes(std::vector<Shape2d>& shapes){
	uint64_t didMerge = 0;
	std::vector<Shape2d> merged;
	merged.reserve(shapes.size());
	merged.push_back(std::move(shapes[0]));
	for(uint32_t i = 1; i < shapes.size(); ++i){
		bool mergedWithExisting = false;
		for(auto& existing : merged){
			if(existing.shapes_overlap(shapes[i])){
				existing.merge_shape(std::move(shapes[i]));
				mergedWithExisting = true;
				++didMerge;
				break;
			}
		}
		if(!mergedWithExisting){merged.push_back(std::move(shapes[i]));}
	}
	shapes = std::move(merged);
	return didMerge;
}

//for the very first one, going to start with a radius of (0,1)
void Refine2d::start(){
	std::vector<Point2d> points = _points_hndlr.generatePoints();
	std::vector<Shape2d> cur_shapes;
	cur_shapes.reserve(points.size());
	std::cout << "making cur shapes from "<<points.size()<<" points"<<std::endl;
	for(uint64_t i = 0; i < points.size(); ++i){
		cur_shapes.push_back(Shape2d(_points_hndlr.in_radius_one(points[i])));
		utils::printProgressBar(i, points.size(), "num points: "+std::to_string(cur_shapes[i].points.size()));
	}
	uint64_t merged = mergeAllShapes(cur_shapes);
	while(merged > 0){
		std::cout << "merged is currently "<<merged<<std::endl;
		merged = mergeAllShapes(cur_shapes);
	}
	last_radius = 1;
	for(uint32_t i = 0; i < cur_shapes.size(); ++i){
		std::cout<<i<<": "<<cur_shapes[i].points.size()<<std::endl;
	}
	shapes = std::move(cur_shapes);
}


Refine2d::Refine2d(const Tensor& t, const uint8_t point)
	:_points_hndlr(t, point), last_radius(0)
{start();}

Refine2d::Refine2d(Tensor&& t, const uint8_t point)
	:_points_hndlr(std::move(t), point), last_radius(0)
{start();}

inline bool isInside(const Simplex2d& inner, const Simplex2d& outter){
	return (inner[0].first <= outter[0].first
			&& inner[0].second <= outter[0].second)
		&& (inner[1].first <= outter[1].first
			&& inner[1].second <= outter[1].second)
		&& (inner[2].first <= outter[2].first
			&& inner[2].second <= outter[2].second);
 
 
}

inline bool isInsideMapped(const Simplex2d& inner, const Simplex2d& outter){
	return (inner[0].first <= outter[0].first
			&& inner[0].second <= outter[0].second)
		&& (inner[1].first <= outter[1].first
			&& inner[1].second <= outter[1].second)
		&& (inner[2].first <= outter[2].first
			&& inner[2].second <= outter[2].second); 
}

inline bool isSameSimplex(const Simplex2d& a, const Simplex2d& b){
	return (a[0].first == b[0].first && a[0].second == b[0].second
			&& a[1].first == b[1].first && a[1].second == b[1].second
			&& a[2].first == b[2].first && a[2].second == b[2].second);
}



std::vector<Simplex2d> filterSimplexes(const std::vector<Simplex2d>& simps, size_t index){
	std::unordered_map<Point2d, Point2d, Point2dHash> curPairs;
	for(int64_t i = 0; i < simps.size(); ++i){
		const Point2d& key = simps[i][index];
		if(curPairs.find(key) != curPairs.end()){
			curPairs[key].second = i;
			continue;
		}
		curPairs[key] = Point2d(i,i);
	}
	std::vector<Simplex2d> outp;
#ifdef USE_PARALLEL
	tbb::spin_mutex mutex;
	std::atomic_int64_t check;
	check.store(static_cast<int64_t>(0));
	tbb::parallel_for(tbb::blocked_range<size_t>(0, simps.size()),
		[&](const tbb::blocked_range<size_t>& range){
		check.fetch_add(range.end()-range.begin(), std::memory_order_relaxed);
		utils::printThreadingProgressBar(check.load(), simps.size());
		auto begin = simps.cbegin() + range.begin();
		auto end = simps.cbegin() + range.end();
		std::vector<Simplex2d> mOutp;
		mOutp.reserve(range.end()-range.begin());
		for(;begin != end; ++begin){
			const Point2d& val = curPairs[(*begin)[index]];
			const int64_t& start = val.first;
			const int64_t& end = val.second;
			bool inside = false;
			for(int64_t i = start; i <= end; ++i){
				if(isSameSimplex(*begin, simps[i]))
					continue;
				if(isInsideMapped(*begin, simps[i])){
					inside = true;
					break;
				}
			}
			if(!inside){mOutp.push_back(*begin);}
		}
		tbb::spin_mutex::scoped_lock lock(mutex);
		outp.reserve(mOutp.size()+1);
		outp.insert(outp.end(), mOutp.begin(), mOutp.end());
		lock.release();
		});
#else
	uint64_t i =0;
	const uint64_t total = simps.size();
	for(const Simplex2d& tSimplex : simps){
		utils::printProgressBar(i, total);
		bool inside = false;
		for(const Simplex2d& nSimplex : simps){
			if(isSameSimplex(tSimplex, nSimplex))
				continue;
			if(isInside(tSimplex, nSimplex)){
				inside = true;
				break;
			}
		}
		if(!inside){outp.push_back(tSimplex);}
		++i;
	}
#endif
	return std::move(outp);
}

double triangleArea(const Point2d& p1, const Point2d& p2, const Point2d& p3) {
    return std::abs(0.5 * (p1.first * (p2.second - p3.second) +
                            p2.first * (p3.second - p1.second) +
                            p3.first * (p1.second - p2.second)));
}

// Function to generate all integer points within a simplex
unordered_point2d_set generatePointsWithinSimplex(const Simplex2d& simplex) {
    unordered_point2d_set points;

    // Get the bounding box of the simplex
    int64_t minX = std::min({simplex[0].first, simplex[1].first, simplex[2].first});
    int64_t minY = std::min({simplex[0].second, simplex[1].second, simplex[2].second});
    int64_t maxX = std::max({simplex[0].first, simplex[1].first, simplex[2].first});
    int64_t maxY = std::max({simplex[0].second, simplex[1].second, simplex[2].second});

    // Iterate over each point in the bounding box
    for (int64_t x = minX; x <= maxX; ++x) {
        for (int64_t y = minY; y <= maxY; ++y) {
            Point2d p(x, y);
            // Calculate the areas of the three triangles formed by the point and the simplex vertices
            double totalArea = triangleArea(simplex[0], simplex[1], simplex[2]);
            double area1 = triangleArea(p, simplex[0], simplex[1]);
            double area2 = triangleArea(p, simplex[1], simplex[2]);
            double area3 = triangleArea(p, simplex[2], simplex[0]);
            // If the sum of the areas equals the area of the simplex, the point is inside the simplex
            if (std::abs(totalArea - (area1 + area2 + area3)) < 1e-9) {
                points.insert(p);
            }
        }
    }

    return points;
}


// Function to generate all integer points on a line between two points (Bresenham's line algorithm)
std::vector<Point2d> generatePointsOnLine(const Point2d& p1, const Point2d& p2) {
    std::vector<Point2d> points;

    // Difference between the two points
    int64_t dx = p2.first - p1.first;
    int64_t dy = p2.second - p1.second;
    points.reserve(dx);

    // Direction of the line
    int64_t sx = (dx > 0) ? 1 : -1;
    int64_t sy = (dy > 0) ? 1 : -1;

    // Absolute differences
    dx = std::abs(dx);
    dy = std::abs(dy);

    // Error in the decision variable
    int64_t err = dx - dy;

    // Starting point
    int64_t x = p1.first;
    int64_t y = p1.second;

    while (true) {
        // Add current point to the list
        points.push_back({x, y});

        // Check if reached the end point
        if (x == p2.first && y == p2.second) {
            break;
        }

        // Calculate the next point
        int64_t e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }

    return points;
}


//this is for when sortedPoints[0].second == sortedPoints[1].second or sortedPoints[1].second == sortedPoints[2].second
void generatePointsInHorizontalSimplex(const Point2d& p1, const Point2d& p2, const Point2d& p3, unordered_point2d_set& points) {

    // Sort points by y-coordinate
    std::vector<Point2d> sortedPoints = {p1, p2, p3};
    std::sort(sortedPoints.begin(), sortedPoints.end(), [](const Point2d& a, const Point2d& b) {
        return a.second < b.second;
    });

    // Find the horizontal edge
    Point2d horizontalPoint1, horizontalPoint2, vertex;
    if (sortedPoints[0].second == sortedPoints[1].second) {
        horizontalPoint1 = sortedPoints[0];
        horizontalPoint2 = sortedPoints[1];
        vertex = sortedPoints[2];
    } else if (sortedPoints[1].second == sortedPoints[2].second) {
        horizontalPoint1 = sortedPoints[1];
        horizontalPoint2 = sortedPoints[2];
        vertex = sortedPoints[0];
    } else {
        // Not a valid simplex configuration
        return;
    }

    //this could probably skip the generate points on the horizontal line function if it is slow
    //it would be just as easy to go with the min and max, since they will have the same y, just generate by x
    // Generate points on the horizontal edge
    std::vector<Point2d> horizontalPoints = generatePointsOnLine(horizontalPoint1, horizontalPoint2);

    // Iterate over each point on the horizontal edge and fill in the triangle
    for (const Point2d& p : horizontalPoints) {
        // Calculate x-range within the triangle at this y-coordinate
        int64_t x1 = horizontalPoint1.first + (p.second - horizontalPoint1.second) * (vertex.first - horizontalPoint1.first) / (horizontalPoint2.second - horizontalPoint1.second);
        int64_t x2 = horizontalPoint1.first + (p.second - horizontalPoint1.second) * (vertex.first - horizontalPoint1.first) / (horizontalPoint2.second - horizontalPoint1.second);

        // Add points within the x-range
        for (int64_t x = x1; x <= x2; ++x) {
            points.insert({x, p.second});
        }
    }

}


// Compute Barycentric coordinates, assuming (1/demon) is already included
bool isPointInSimplex(const Point2d& point, const Simplex2d& simplex, double demon){
	double u, v, w;
	u = ((simplex[1].second - simplex[2].second) * (point.first - simplex[2].first) +
		(simplex[2].first - simplex[1].first) * (point.second - simplex[2].second)) * demon;
	
	v = ((simplex[2].second - simplex[0].second) * (point.first - simplex[2].first) +
         (simplex[0].first - simplex[2].first) * (point.second - simplex[2].second)) * demon;
	w = 1.0 - u - v;
	return (u >= 0 && v >= 0 && w >= 0);
}


unordered_point2d_set generatePointsWithinSimplex(const std::vector<std::reference_wrapper<const Simplex2d> >& simplexes){
	unordered_point2d_set points;
	int64_t minX = simplexes[0].get()[0].first;
	int64_t maxX = simplexes[0].get()[0].first;
	int64_t minY = simplexes[0].get()[0].second;
	int64_t maxY = simplexes[0].get()[0].second;
	for (const auto& simplex : simplexes) {
		minX = std::min({minX, simplex.get()[0].first, simplex.get()[1].first, simplex.get()[2].first});
		maxX = std::max({maxX, simplex.get()[0].first, simplex.get()[1].first, simplex.get()[2].first});
		minY = std::min({minY, simplex.get()[0].second, simplex.get()[1].second, simplex.get()[2].second});
		maxY = std::max({maxY, simplex.get()[0].second, simplex.get()[1].second, simplex.get()[2].second});
	}

	std::vector<double> divs;
	divs.reserve(simplexes.size());
	for(const auto& simp : simplexes){
		const Simplex2d& simplex = simp.get();
		double denom = static_cast<double>((simplex[1].second - simplex[2].second) * (simplex[0].first - simplex[2].first) + (simplex[2].first - simplex[1].first) * (simplex[0].second - simplex[2].second));
		if(denom == 0){
			if(simplex[0].first == simplex[1].first && simplex[1].first == simplex[2].first){
				int64_t min_y = std::min(simplex[2].second, std::min(simplex[0].second, simplex[1].second));
				int64_t max_y = std::max(simplex[2].second, std::max(simplex[0].second, simplex[1].second));
				const int64_t& cur_x = simplex[0].first;
				for(int64_t y = min_y; y <= max_y; ++y){
					points.insert(Point2d(cur_x, y));
				}
			}
			else if(simplex[1].second == simplex[2].second && simplex[0].second == simplex[1].second){
				int64_t min_x = std::min(simplex[2].first, std::min(simplex[0].first, simplex[1].first));
				int64_t max_x = std::max(simplex[2].first, std::max(simplex[0].first, simplex[1].first));
				const int64_t& cur_y = simplex[0].second;
				for(int64_t x = min_x; x <= max_x; ++x){
					points.insert(Point2d(x, cur_y));
				}
			}
			else{
				generatePointsInHorizontalSimplex(simplex[0], simplex[1], simplex[2], points);
			}
			divs.push_back(0);
			continue;
		}
		divs.push_back(1/denom);
	}
	for(int64_t x = minX; x <= maxX; ++x){
		for(int64_t y = minY; y <= maxY; ++y){
			Point2d point(x, y);
			if(points.find(point) != points.end())
				continue;
			for(uint32_t i = 0; i < simplexes.size(); ++i){
				if(isPointInSimplex(point, simplexes[i].get(), divs[i])){
					points.insert(point);
					break;
				}
			}
		}
	}

	/* std::vector<double> triangleAreas; */
	/* triangleAreas.reserve(simplexes.size()); */
	/* for(const auto& simplex : simplexes){ */
	/* 	triangleAreas.push_back(triangleArea(simplex.get()[0], simplex.get()[1], simplex.get()[2])); */
	/* } */
	/* for(int64_t x = minX; x <= maxX; ++x){ */
	/* 	for(int64_t y = minY; y <= maxY; ++y){ */
	/* 		Point2d point(x, y); */
	/* 		for(uint32_t i = 0; i < simplexes.size(); ++i){ */
	/* 			const double& totalArea = triangleAreas[i]; */
	/* 			const Simplex2d& simplex = simplexes[i].get(); */
	/* 			double area1 = triangleArea(point, simplex[0], simplex[1]); */
	/* 			double area2 = triangleArea(point, simplex[1], simplex[2]); */
	/* 			double area3 = triangleArea(point, simplex[2], simplex[0]); */
	/* 			if(std::abs(totalArea - (area1 + area2 + area3)) < 1e-9){ */
	/* 				points.insert(point); */
	/* 				break; */
	/* 			} */
	/* 		} */
	/* 	} */
	/* } */
	return std::move(points);
}

void generatePointsWithinSimplex(const std::vector<std::reference_wrapper<const Simplex2d> >& simplexes, Shape2d& shape){
	unordered_point2d_set& points = shape.points;
	int64_t minX = simplexes[0].get()[0].first;
	int64_t maxX = simplexes[0].get()[0].first;
	int64_t minY = simplexes[0].get()[0].second;
	int64_t maxY = simplexes[0].get()[0].second;
	for (const auto& simplex : simplexes) {
		minX = std::min({minX, simplex.get()[0].first, simplex.get()[1].first, simplex.get()[2].first});
		maxX = std::max({maxX, simplex.get()[0].first, simplex.get()[1].first, simplex.get()[2].first});
		minY = std::min({minY, simplex.get()[0].second, simplex.get()[1].second, simplex.get()[2].second});
		maxY = std::max({maxY, simplex.get()[0].second, simplex.get()[1].second, simplex.get()[2].second});
	}
	
	std::vector<double> divs;
	divs.reserve(simplexes.size());
	for(const auto& simp : simplexes){
		const Simplex2d& simplex = simp.get();
		double denom = static_cast<double>((simplex[1].second - simplex[2].second) * (simplex[0].first - simplex[2].first) + (simplex[2].first - simplex[1].first) * (simplex[0].second - simplex[2].second));
		if(denom == 0){
			if(simplex[0].first == simplex[1].first && simplex[1].first == simplex[2].first){
				int64_t min_y = std::min(simplex[2].second, std::min(simplex[0].second, simplex[1].second));
				int64_t max_y = std::max(simplex[2].second, std::max(simplex[0].second, simplex[1].second));
				const int64_t& cur_x = simplex[0].first;
				for(int64_t y = min_y; y <= max_y; ++y){
					points.insert(Point2d(cur_x, y));
				}
			}
			else if(simplex[1].second == simplex[2].second && simplex[0].second == simplex[1].second){
				int64_t min_x = std::min(simplex[2].first, std::min(simplex[0].first, simplex[1].first));
				int64_t max_x = std::max(simplex[2].first, std::max(simplex[0].first, simplex[1].first));
				const int64_t& cur_y = simplex[0].second;
				for(int64_t x = min_x; x <= max_x; ++x){
					points.insert(Point2d(x, cur_y));
				}
			}
			else{
				generatePointsInHorizontalSimplex(simplex[0], simplex[1], simplex[2], points);
			}
			divs.push_back(0);
			continue;
		}
		divs.push_back(1/denom);
	}
	for(int64_t x = minX; x <= maxX; ++x){
		for(int64_t y = minY; y <= maxY; ++y){
			Point2d point(x, y);
			if(points.find(point) != points.end())
				continue;
			for(uint32_t i = 0; i < simplexes.size(); ++i){
				if(isPointInSimplex(point, simplexes[i].get(),divs[i])){
					points.insert(point);
					break;
				}
			}
		}
	}


	//below is the old way it was done, instead I am going to use Barycentric coordinates
	/* std::vector<double> triangleAreas; */
	/* triangleAreas.reserve(simplexes.size()); */
	/* for(const auto& simplex : simplexes){ */
	/* 	triangleAreas.push_back(triangleArea(simplex.get()[0], simplex.get()[1], simplex.get()[2])); */
	/* } */
	/* for(int64_t x = minX; x <= maxX; ++x){ */
	/* 	for(int64_t y = minY; y <= maxY; ++y){ */
	/* 		Point2d point(x, y); */
	/* 		if(points.find(point) != points.end()) */
	/* 			continue; */
	/* 		for(uint32_t i = 0; i < simplexes.size(); ++i){ */
	/* 			const double& totalArea = triangleAreas[i]; */
	/* 			const Simplex2d& simplex = simplexes[i].get(); */
	/* 			double area1 = triangleArea(point, simplex[0], simplex[1]); */
	/* 			double area2 = triangleArea(point, simplex[1], simplex[2]); */
	/* 			double area3 = triangleArea(point, simplex[2], simplex[0]); */
	/* 			if(std::abs(totalArea - (area1 + area2 + area3)) < 1e-9){ */
	/* 				points.insert(point); */
	/* 				break; */
	/* 			} */
	/* 		} */
	/* 	} */
	/* } */

}

void generate_and_merge(std::vector<Simplex2d>& simps, std::vector<Shape2d>& shapes, size_t index=0){
	std::unordered_map<Point2d, std::vector<std::reference_wrapper<const Simplex2d> >, Point2dHash> Map;
	for(const Simplex2d& simp : simps){
		if(Map.find(simp[index]) != Map.end()){
			Map[simp[index]].emplace_back(std::cref(simp));
			continue;
		}
		Map[simp[index]] = std::vector<std::reference_wrapper<const Simplex2d>>(1, std::cref(simp));
		Map[simp[index]].reserve(20);
	}
	std::cout << "made "<<Map.size()<<" groupings"<<std::endl;
	uint64_t mergedCntr = 0;
	for(const auto& pair : Map){
		bool merged = false;
		for(uint32_t i = 0; i < shapes.size(); ++i){
			if(shapes[i].points.find(pair.first) != shapes[i].points.end()){
				generatePointsWithinSimplex(pair.second, shapes[i]);
				merged = true;
				++mergedCntr;
				break;
			}
		}
		if(!merged){shapes.emplace_back(Shape2d(generatePointsWithinSimplex(pair.second)));}
	}
	std::cout << "finished merging "<<mergedCntr<<" shapes"<<std::endl;
	
}


void Refine2d::increment_radius(int64_t adding){
	simplexes_2d si(_points_hndlr, last_radius + adding, last_radius, ::nt::tda::detail::FilterSimplexes{});
	const std::vector<Simplex2d>& simps = si.getSimplexes();
	std::cout << "made simplexes, now filtering..."<<std::endl;
	

	std::vector<Simplex2d> nSimps = filterSimplexes(simps, 0);
	std::cout <<std::endl<< "filtered "<<simps.size()<<" simplexes down to "<<nSimps.size()<<std::endl;
	generate_and_merge(nSimps, shapes);
	uint64_t merged = mergeAllShapes(shapes);
	while(merged > 0){
		std::cout << "merged is currently "<<merged<<std::endl;
		merged = mergeAllShapes(shapes);
	}
	std::cout << "finished incrementing"<<std::endl;
	last_radius += adding;
}

void Refine2d::save_current_step(const std::string out_dir) const{
	nt::Tensor to_save = nt::functional::zeros(_points_hndlr.original_tensor().shape(), shapes.size() < 200 ? DType::uint8 : DType::uint32);
	for(uint32_t i = 0; i < shapes.size(); ++i){
		utils::printProgressBar(i, shapes.size());
		shapes[i].fill_tensor(to_save, Scalar(i+1));
	}
	std::cout << std::endl;
	std::vector<std::array<uint8_t, 3>> colors = generateColors(shapes.size());
	nt::Tensor out_img = nt::functional::zeros({3, to_save.shape()[-2], to_save.shape()[-1]}, nt::DType::uint8);
	uint8_t* r_begin = reinterpret_cast<uint8_t*>(out_img.data_ptr());
	uint8_t* g_begin = r_begin + (to_save.shape()[-2] * to_save.shape()[-1]);
	uint8_t* b_begin = g_begin + (to_save.shape()[-2] * to_save.shape()[-1]);
	if(to_save.dtype == DType::uint8){
		uint8_t* begin = reinterpret_cast<uint8_t*>(to_save.data_ptr());
		uint8_t* end = begin + (to_save.shape()[-2] * to_save.shape()[-1]);
		for(;begin != end; ++begin, ++r_begin, ++g_begin, ++b_begin){
			*r_begin = colors[*begin][0];
			*g_begin = colors[*begin][1];
			*b_begin = colors[*begin][2];
		}
	}
	else{
		uint32_t* begin = reinterpret_cast<uint32_t*>(to_save.data_ptr());
		uint32_t* end = begin + (to_save.shape()[-2] * to_save.shape()[-1]);
		for(;begin != end; ++begin, ++r_begin, ++g_begin, ++b_begin){
			*r_begin = colors[*begin][0];
			*g_begin = colors[*begin][1];
			*b_begin = colors[*begin][2];
		}	
	}
	images::Image img;
	img.savePPM(out_dir + "/image_"+std::to_string(last_radius)+".ppm", out_img);

	
}

}
}
}
