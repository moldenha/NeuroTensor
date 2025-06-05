#ifndef _NT_OLD_TDA_SHAPES_H_
#define _NT_OLD_TDA_SHAPES_H_
#include "../../functional/functional.h"
#include "../../Tensor.h"
#include "../../utils/utils.h"
#include <array>
#include "Points.h"
#include "Simplex.h"
#include "Basis.h"
#include <utility>
#include <unordered_set>
#include "../../images/image.h"


namespace nt{
namespace tda{




struct shape_2d{
	std::vector<Simplex2d> shape;
	inline const size_t simplex_amount() const {return shape.size();}
	inline void add_simplex(const Simplex2d& simplex){
		shape.emplace_back(simplex);
		pairSet.insert(simplex[0]);
		pairSet.insert(simplex[1]);
		pairSet.insert(simplex[2]);
	}
	inline bool simplex_in_shape(const Simplex2d& simplex) const{
		return (pairSet.find(simplex[0]) != pairSet.end()) ||  
			(pairSet.find(simplex[1]) != pairSet.end()) ||
			(pairSet.find(simplex[2]) != pairSet.end());
	}
	inline bool simplex_already_added(const Simplex2d& simplex) const{
		return (pairSet.find(simplex[0]) != pairSet.end()) && 
			(pairSet.find(simplex[1]) != pairSet.end()) &&
			(pairSet.find(simplex[2]) != pairSet.end());
		
	}
	inline bool check_and_add(const Simplex2d& simplex) {
		if(!simplex_already_added(simplex) && simplex_in_shape(simplex)){
			add_simplex(simplex); return true;} 
		return false;
	}
	shape_2d(Simplex2d simp)
		:shape(1)
	{shape[0] = simp;
	pairSet.insert(simp[0]);
	pairSet.insert(simp[1]);
	pairSet.insert(simp[2]);}

	bool shapes_overlap(const shape_2d&) const;
	void merge_shape(shape_2d&&);

	shape_2d(std::vector<Simplex2d> simp)
		:shape(std::move(simp))
	{
	for(uint32_t i = 0; i < simp.size(); ++i){
		pairSet.insert(simp[i][0]);
		pairSet.insert(simp[i][1]);
		pairSet.insert(simp[i][2]);
	}
	}

	shape_2d(std::vector<Simplex2d> simp, unordered_point2d_set pairs)
		:shape(std::move(simp)), pairSet(std::move(pairs))
	{}
	double area() const;
	void fill_tensor(Tensor&, Scalar) const;
	void fill_tensor(Tensor&, Scalar, const std::vector<Simplex2d >&) const;
	private:
		unordered_point2d_set pairSet;
};

class shapes_2d{
	std::vector<shape_2d> shapes;
	std::vector<shape_2d> generateShapes(const simplexes_2d&) const;
	public:
		shapes_2d() = delete;
		shapes_2d(const simplexes_2d&);
		inline const std::vector<shape_2d>& getShapes() const {return shapes;}
		shapes_2d& combine_self(shapes_2d&&);
};


struct Shape{
	BasisOverlapping balls; //hold all the balls of the simplexes
	Simplexes simps;
	Shape() = delete;
	Shape(const BasisOverlapping& bs)
		:balls(bs),
		simps(Simplexes(bs))
	{}	
};


class Shapes{
	std::vector<Shape> shapes;
	Basises Balls;
	inline void generateShapes(){
		const auto& balls = Balls.getBalls();

		shapes.clear();
		shapes.reserve(balls.size());
		for(const auto& ball : balls){
			shapes.push_back(Shape(ball));
		}
	}
	public:
		explicit Shapes(Points p)
			:Balls(Basises(p))
		{}

		inline void setRadius(double r, bool generate_shapes=true){
			Balls.radius_to(r);
			if(generate_shapes){generateShapes();}
		}
		inline const Basis& getBasis(const Point& p){return Balls.getBasis(p);}
		inline const BasisOverlapping& getBasisOverlapping(const Point& p){return Balls[p];}
		inline const Basises& getBalls() const noexcept {return Balls;}
		inline const std::vector<Shape>& getShapes() const noexcept {return shapes;}
		inline void adjust_radius(double r, bool generate_shapes=true){
			Balls.adjust_radius(r);
			if(generate_shapes){generateShapes();}
		}
};

inline void set(uint8_t* ptr, const uint32_t& rows, const uint32_t& cols, const Point& point, uint8_t setting) noexcept {
	/* utils::THROW_EXCEPTION(point.size() == 2, "Expected point to have a size of 2 but had a size of $".format(point.size())); */
	if(point[0] < 0 || point[0] >= rows)
		return;
	if(point[1] < 0 || point[1] >= cols)
		return;
	ptr[point[0] * cols + point[1]] = setting;
}


//assumes tensor is dtype uint8_t
inline void fillTensor(Tensor& tensor, const Point& center, uint8_t setting, double radius){
	utils::throw_exception(center.dims() == 2, "Expected to fill tensor with a basis dimension of 2 but got $", center.dims());
	utils::throw_exception(tensor.dtype == DType::uint8, "Expected to get a dtype for fill tensor of $ but got $", DType::uint8, tensor.dtype);
	uint8_t* ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr());
	Tensor t({1}, DType::uint8);
	t = 1;
	Points pts(std::move(t), 1);
	unordered_point_set s = pts.generate_all_points_within_radius(center, 0, radius);
	set(ptr, tensor.shape()[-2], tensor.shape().back(), center, setting);
	for(const Point& point : s){
		set(ptr, tensor.shape()[-2], tensor.shape()[-1], point, setting);
	}
}


inline std::vector<Point> generate_points_on_line(const Point& p1, const Point& p2) {
    utils::throw_exception(p1.size() == p2.size() && p1.size() == 2, "Expected to draw lines between 2d points, but gtot $ and $", p1.size(), p2.size());
    int64_t x1, y1, x2, y2;
    x1 = p1[0];
    y1 = p1[1];
    x2 = p2[0];
    y2 = p2[1];

    std::vector<Point> points;

    int64_t dx = std::abs(x2 - x1);
    int64_t dy = std::abs(y2 - y1);
    int64_t sx = (x1 < x2) ? 1 : -1;
    int64_t sy = (y1 < y2) ? 1 : -1;
    int64_t err = dx - dy;

    while (true) {
        points.push_back(Point({x1, y1}));
        if (x1 == x2 && y1 == y2) break;
        int64_t e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }

    return std::move(points);
}

inline void drawLine(Tensor& grid, const std::vector<Point>& points) {
	
    uint8_t* ptr = reinterpret_cast<uint8_t*>(grid.data_ptr());
    for(const auto& point : points){
	ptr[point[0] * grid.shape().back() + point[1]] = 0;
    }
}




void visualize(const Points& points, uint32_t height, uint32_t width, uint32_t channels, int64_t point_radius=10, int64_t ball_radius=50, bool only_biggest_shape=false, uint32_t shapes_above=0);

inline void visualize(const Points& points, uint32_t height, uint32_t width, std::string output, int64_t point_radius = 20, int64_t ball_radius=100){
        utils::throw_exception(points.dims() == 2, "Expected to visualize 2d points, but got $d", points.dims());
	//steps:
	//(1) get a ball around each point so it can be visualized
	std::cout<<"getting red points"<<std::endl;
	Basises redPoints(points);
	redPoints.radius_to(point_radius);
	//(2) get all the basises that are connected to each other
	std::cout << "getting basis's"<<std::endl;
	Shapes shapes(points);
	std::cout<<"setting radius"<<std::endl;
	shapes.setRadius(ball_radius);
	std::cout<<"set radius"<<std::endl;
	//(3) make a tensor that is the correct width and height
	std::cout << "getting output filling with one"<<std::endl;
	Tensor outp = ::nt::functional::ones({height, width}, DType::uint8);
	std::cout<<"got outp"<<std::endl;
	//(4) get all the points and visualize the balls
	std::cout << "getting all basis's and filling tensor"<<std::endl;
	std::vector<Point> pts = points.generatePoints();
	for(const auto& point : pts){
		const Basis& basis = shapes.getBasis(point);
		fillTensor(outp, basis.center, 2, basis.radius);
		
	}
	//next visualize all the points on top of the basis's
	std::cout << "filling tensor with red point"<<std::endl;
	for(const auto& point : pts){
		const Basis& basis = redPoints.getBasis(point);
		fillTensor(outp, basis.center, 3, basis.radius);
	}
	//next vistualize all the points that are connected
	std::cout << "drawing lines"<<std::endl;
	for(const auto& point : pts){
		const BasisOverlapping& overlap = shapes.getBasisOverlapping(point);
		std::vector<Basis> connected = overlap.getConnected(point);
		for(const auto& basis : connected){
			auto vec = generate_points_on_line(point, basis.center);
			drawLine(outp, vec);
			/* drawLine(outp, point, basis.center); */
		}
	}
	std::cout << "making output"<<std::endl;
	//last output it:
	Tensor img = ::nt::functional::zeros({3, height, width}, DType::uint8);
	uint8_t* R = reinterpret_cast<uint8_t*>(img.data_ptr());
	uint8_t* G = R + (height * width);
	uint8_t* B = G + (height * width);
	uint8_t* begin = reinterpret_cast<uint8_t*>(outp.data_ptr());
	uint8_t* end = reinterpret_cast<uint8_t*>(outp.data_ptr_end());
	for(;begin != end; ++begin, ++R, ++G, ++B){
		if(*begin == 0){
			*R = 0;
			*G = 0;
			*B = 0;
		}
		else if(*begin == 1){
			*R = 255;
			*G = 255;
			*B = 255;
		}
		else if(*begin == 2){
			*R = 0;
			*G = 0;
			*B = 255;
		}
		else if(*begin == 3){
			*R = 255;
			*G = 0;
			*B = 0;
		}
	}
	images::Image toImg;
	toImg.savePPM(output, img);

}

}
}

#endif //_NT_TDA_SHAPES_H_
