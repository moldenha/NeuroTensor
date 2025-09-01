#ifndef NT_IMAGE_PROCESSOR_H__
#define NT_IMAGE_PROCESSOR_H__

#include "../Tensor.h"
#include <vector>

namespace nt{
namespace images{

class NEUROTENSOR_API Image{
	Tensor pixels;
	Tensor read_img(const char*, DType) const;
	Tensor read_stb(const char*, DType) const;
	Tensor read_ppm(const char*, DType) const;
	public:
		Image();
		Image(const char*, DType dt = DType::Float);
		Image(const Tensor&);
		Image(Tensor&&);
		Image(const Image&);
		Image(Image&&);

		inline Image& operator=(Image&& img){pixels = std::move(img.pixels); return *this;}
		inline Image& operator=(const Image& img){pixels = img.pixels; return *this;}
		inline Image& operator=(const Tensor& p){pixels = p; return *this;}
		inline Image& operator=(Tensor&& p){pixels = std::move(p); return *this;}
		inline Tensor& pix() {return pixels;}
		inline const Tensor& pix() const {return pixels;}
		inline void save(const std::string& filename) const {saveImage(filename, pixels);}
		void savePPM(const std::string& filename, const Tensor& rgb) const;
		void saveSTBI(const std::string& filename, const Tensor& rgb) const;
		void saveImage(const std::string& filename, const Tensor& rgb) const;




		

};


}
}

#endif //_NT_IMAGE_PROCESSOR_H_ 
