#include "n_refine.h"
#include "../Points.h"
#include "../Simplex.h"
#include "../Shapes.h"
#include <vector>
#include <unordered_set>
#include <utility>

namespace nt{
namespace tda{
namespace refine{


Point crossProduct(const Point& u, const Point& v) {
    assert(u.size() == 3 && v.size() == 3);
    return {
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    };
}

// Helper function to compute the dot product of two vectors
int64_t dotProduct(const Point& u, const Point& v) {
    assert(u.size() == v.size());
    int64_t sum = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        sum += u[i] * v[i];
    }
    return sum;
}

// Helper function to subtract two vectors
Point subtract(const Point& a, const Point& b) {
    assert(a.size() == b.size());
    Point result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Function to check if a point is within a 3D simplex
bool isPointInSimplex(const Point& point, const Simplex& simplex) {
    assert(simplex.size() == 4);  // Simplex should be a tetrahedron

    // Function to check if a point is on the same side of a plane
    auto isSameSide = [](const Point& p1, const Point& p2, const Point& a, const Point& b, const Point& c) {
        Point ab = subtract(b, a);
        Point ac = subtract(c, a);
        Point ap1 = subtract(p1, a);
        Point ap2 = subtract(p2, a);

        Point normal = crossProduct(ab, ac);
        return dotProduct(normal, crossProduct(ab, ap1)) >= 0 && dotProduct(normal, crossProduct(ab, ap2)) >= 0;
    };

    const Point& a = simplex[0];
    const Point& b = simplex[1];
    const Point& c = simplex[2];
    const Point& d = simplex[3];

    // Check if the point is on the same side of all tetrahedron faces
    return isSameSide(point, a, b, c, d) &&
           isSameSide(point, b, a, c, d) &&
           isSameSide(point, c, a, b, d) &&
           isSameSide(point, d, a, b, c);
}

// Helper function to compute the area of a triangle given three points
int64_t area(const Point2D& a, const Point2D& b, const Point2D& c) {
    return (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2;
}

// Function to check if a point is within a 2D triangle
bool isPointInTriangle(const Point2D& point, const Triangle& triangle) {
    assert(triangle.size() == 3);  // Triangle should have exactly 3 points

    const Point2D& a = triangle[0];
    const Point2D& b = triangle[1];
    const Point2D& c = triangle[2];

    // Compute the area of the full triangle and the areas of the sub-triangles
    int64_t A = area(a, b, c);
    int64_t A1 = area(point, b, c);
    int64_t A2 = area(a, point, c);
    int64_t A3 = area(a, b, point);

    // The point is inside the triangle if the sum of the areas of the sub-triangles equals the area of the full triangle
    return (A == A1 + A2 + A3);
}


inline const BasisOverlapping& get_biggest_shape(const Shapes& shape_hndlr) noexcept{
	const std::vector<BasisOverlapping>& balls = shape_hndlr.getBalls().getBalls();
	uint32_t index = 0;
	for(uint32_t i = 1; i < balls.size(); ++i){
		if(balls[i].points.size() > balls[index].points.size())
			index = i;
	}
	return balls[index];
}


inline void set_2d_tensor_point(uint8_t* out, const Point& p, const int64_t& rows, const int64_t& cols) noexcept {
	if(p[0] > rows || p[1] > cols)
		return;
	out[p[0] * cols + p[1]] = 1;
}

inline void set_nd_tensor_point(uint8_t* out, const Point& p, const SizeRef& t_shape) noexcept {
	int64_t index = 0;
	for(int64_t i = 0; i < p.size(); ++i){
		if(p[i] > t_shape[i]){return;}
		index += (p[i] * t_shape.multiply(i+1));
	}
	out[index] = 1;
}

inline void set_tensor_points_r1(Tensor& out, const BasisOverlapping& Balls, bool is_2d){
	if(is_2d){
	utils::THROW_EXCEPTION(out.dims() == 2, "when generating refinements, when setting is_2d = true, expected input tensor dims to be 2 but got $", out.dims());
	}
	utils::THROW_EXCEPTION(out.dtype == DType::uint8, "Internal logic error");
	uint8_t* data = reinterpret_cast<uint8_t*>(out.data_ptr());
	if(is_2d){
		const int64_t& rows = out.shape()[-2];
		const int64_t& cols = out.shape()[-1];
		for(const auto& point : Balls.points)
			set_2d_tensor_point(data, point, rows, cols);
		return;
	}
	const SizeRef& t_shape = out.shape();
	for(const auto& point : Balls.points)
		set_nd_tensor_point(data, point, t_shape);
}


//this is going to build on itself and generate the biggest shape in the original tensor
std::vector<Tensor> generateRefinements(const Tensor& _t, std::vector<double> radi, uint8_t point, bool is_2d){
	if(is_2d){utils::throw_exception(_t.dims() == 2, "when generating refinements, when setting is_2d = true, expected input tensor dims to be 2 but got $", _t.dims());}
	if(!is_2d && _t.dims() == 2){is_2d = true;}
	Tensor t = _t.to_dtype(DType::uint8);
	Points points(t, point, (is_2d) ? 2 : t.dims());
	if(is_2d){
		utils::throw_exception(points.dims() == 2, "Was unable to make 2d point set");
	}
	Shapes shape_hndlr(points);
	std::vector<Tensor> output;

	Tensor output_n = functional::zeros_like(t);
	std::cout << "setting radius 1 ...";
	shape_hndlr.setRadius(1.0);
	set_tensor_points_r1(output_n, get_biggest_shape(shape_hndlr), is_2d);
	output.push_back(output_n.clone());
	std::cout << "done"<<std::endl;

}

}}} //nt::tda::refine::
