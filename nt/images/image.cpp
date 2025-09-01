#include "image.h"
#include "../Tensor.h"
#include "../dtype/ArrayVoid.hpp"
#include "../utils/utils.h"
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <stb_sprintf.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// #include "../../third_party/matplot/source/3rd_party/cimg/CImg.h"

// #ifdef cimg_use_jpeg
// #error "CIMG jpeg defined"
// #else
// #error "CIMG jpeg not defined"
// #endif


namespace nt{
namespace images{



Image::Image()
	:pixels({3, 1, 1}, DType::Float)
{}

Image::Image(const char* fname, DType dt)
	:pixels(read_img(fname, dt))
{}

Image::Image(const Tensor& p)
	:pixels(p)
{}

Image::Image(Tensor&& p)
	:pixels(std::move(p))
{}

Image::Image(const Image& img)
	:pixels(img.pixels)
{}

Image::Image(Image&& img)
	:pixels(std::move(img.pixels))
{}


//stb supports JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
Tensor Image::read_img(const char* fname, DType dt) const{
	if(utils::endsWith(fname, ".bmp") ||
		utils::endsWith(fname, ".gif") ||
		utils::endsWith(fname, ".hdr") ||
		utils::endsWith(fname, ".jpg") ||
		utils::endsWith(fname, ".jpeg") ||
		utils::endsWith(fname, ".png") ||
		utils::endsWith(fname, ".psd") ||
		utils::endsWith(fname, ".tga") ||
		utils::endsWith(fname, ".pic") ||
		utils::endsWith(fname, ".ppm") ||
		utils::endsWith(fname, ".pgm") ||
		utils::endsWith(fname, ".pbm") ||
		utils::endsWith(fname, ".pnm")	)
		return read_stb(fname, dt);
	if(utils::endsWith(fname, ".ppm"))
		return read_ppm(fname, dt);
	std::cout << "filetype from "<<fname<<" unsupported"<<std::endl;
	return Tensor({3, 1, 1}, dt);
}


void parsePPM(const std::string& filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string magic, line;
    int maxval;
    file >> magic >> width >> height >> maxval;

    if (magic != "P6") {
        std::cerr << "Invalid PPM file format!" << std::endl;
        return;
    }

    if (maxval != 255) {
        std::cerr << "Unsupported maxval in PPM file! was "<<maxval<<" width was "<<width<<" and height was "<<height << std::endl;
        return;
    }

    // Skip newline character
    std::getline(file, line);
}

void Image::savePPM(const std::string& filename, const Tensor& rgb) const {
    utils::throw_exception(rgb.dtype() == DType::uint8, "Expected tensor to have dtype uint8 but had $", pixels.dtype());
    utils::throw_exception(rgb.is_contiguous(), "Expected tensor to be contiguous");
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    const uint32_t& width = rgb.shape()[-1];
    const uint32_t& height = rgb.shape()[-2];

    // Write PPM header
    file << "P6\n" << width << " " << height << "\n255\n";
	uint32_t imageSize = width * height;
	if(rgb.dims() == 3){
	    const uint8_t* r_begin = reinterpret_cast<const uint8_t*>(rgb.data_ptr());
	    const uint8_t* g_begin = r_begin + (width * height);
	    const uint8_t* b_begin = g_begin + (width * height);

	    for(uint32_t i = 0; i < imageSize; ++i, ++r_begin, ++g_begin, ++b_begin){
		file << *r_begin << *g_begin << *b_begin;
	    }
    }
    else{
	const uint8_t* begin = reinterpret_cast<const uint8_t*>(rgb.data_ptr());
	for(uint32_t i = 0; i < imageSize; ++i, ++begin){
		file << *begin << *begin << *begin;
	}
    }

    // Write pixel data

    file.close();
    std::cout << "PPM file saved successfully: " << filename << std::endl;
}

// Permute the tensor to have the order: rows, cols, num_channels
Tensor adjustTensor(const Tensor& rgb, DType dt){
	return rgb.permute({1,2,0}).to_dtype(dt).contiguous();
}

void Image::saveSTBI(const std::string& filename, const Tensor& rgb) const {

    
    //get image dimensions and number of channels
    const uint32_t num_channels = static_cast<uint32_t>(rgb.shape()[-1]);
    const uint32_t rows = static_cast<uint32_t>(rgb.shape()[-3]);
    const uint32_t cols = static_cast<uint32_t>(rgb.shape()[-2]);
    
    //save the image using stb_image_write.h based on the file format
    int result;
    if (utils::endsWith(filename.c_str(),".png")) {
	Tensor data = adjustTensor(rgb, DType::uint8);
        result = stbi_write_png(filename.c_str(), cols, rows, num_channels, reinterpret_cast<uint8_t*>(data.data_ptr()), cols * num_channels);
    } else if (utils::endsWith(filename.c_str(),".bmp")) {
	Tensor data = adjustTensor(rgb, DType::uint8);
        result = stbi_write_bmp(filename.c_str(), cols, rows, num_channels, reinterpret_cast<uint8_t*>(data.data_ptr()));
    } else if (utils::endsWith(filename.c_str(),".jpg") ||utils::endsWith(filename.c_str(),".jpeg")) {
	Tensor data = adjustTensor(rgb, DType::uint8);
        result = stbi_write_jpg(filename.c_str(), cols, rows, num_channels, reinterpret_cast<uint8_t*>(data.data_ptr()), 90); // Quality set to 90 (can be adjusted)
    } else if (utils::endsWith(filename.c_str(),".tga")) {
	Tensor data = adjustTensor(rgb, DType::uint8);
        result = stbi_write_tga(filename.c_str(), cols, rows, num_channels, reinterpret_cast<uint8_t*>(data.data_ptr()));
    } else if (utils::endsWith(filename.c_str(),".hdr")) {
	Tensor data = adjustTensor(rgb, DType::Float32);
        result = stbi_write_hdr(filename.c_str(), cols, rows, num_channels, reinterpret_cast<float*>(data.data_ptr()));
    } else {
        std::cerr << "Unsupported file format: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    if (result == 0) {
        std::cerr << "Error writing image to file." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Image file saved successfully: " << filename << std::endl;
}


void Image::saveImage(const std::string& filename, const Tensor& rgb) const{
	if(utils::endsWith(filename.c_str(), ".ppm")){
		savePPM(filename, rgb);
		return;
	}
	saveSTBI(filename, rgb);
}

Tensor Image::read_ppm(const char* filename, DType dt) const{
	std::ifstream ifs;
	ifs.open(filename, std::ios::binary); 
	try{
		if (ifs.fail()) { 
			throw("Can't open input file"); 
		}
		std::string header;
		int w, h, b;
		ifs >> header; 
		if (strcmp(header.c_str(), "P6") != 0) throw("Can't read input file");
		ifs >> w >> h >> b;
		std::cout << "w: "<<w << "h: "<<h<<"b: "<<b<<std::endl;
		Tensor output({3, static_cast<unsigned int>(w), static_cast<unsigned int>(h)}, DType::uint8);
		std::cout << "made output "<<output.dims() << output.shape()<<std::endl;
		ifs.ignore(256, '\n'); // skip empty lines in necessary until we get to the binary data
		std::cout << "ignored"<<std::endl;
		Tensor ts = output.split_axis(1);
		std::cout << "split axis"<<std::endl;
		uint8_t* r_begin = reinterpret_cast<uint8_t*>(output.data_ptr());
		uint8_t* g_begin = r_begin + (w * h);
		uint8_t* b_begin = g_begin + (w * h);
		unsigned char pix[3];
		for(uint32_t i = 0; i < (w*h); ++i, ++r_begin, ++g_begin, ++b_begin){
			ifs.read(reinterpret_cast<char*>(pix), 3);
			*r_begin = pix[0];
			*g_begin = pix[1];
			*b_begin = pix[2];
		}

		ifs.close();
		return output.to_dtype(dt);
	}
	catch(const char *err){
		fprintf(stderr, "%s\n", err);
		ifs.close();
	}
	return Tensor({3, 1, 1}, dt);

}


//this supports JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
Tensor Image::read_stb(const char* filename, DType dt) const{
	int rows, cols, num_channels;
	std::cout<<"reding image data"<<std::endl;
	uint8_t* image_data = stbi_load(filename, &rows, &cols, &num_channels, 0);
	if(!image_data){
		std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
		exit(EXIT_FAILURE);
	}
	size_t image_size = rows * cols * num_channels;
	std::cout << "read image size {"<<rows<<','<<cols<<','<<num_channels<<'}'<<std::endl;
	Tensor img_a({static_cast<unsigned int>(rows), static_cast<unsigned int>(cols), static_cast<unsigned int>(num_channels)}, DType::uint8);
	uint8_t* data = reinterpret_cast<uint8_t*>(img_a.data_ptr());
	std::memcpy(data, image_data, image_size);
	img_a.RowColSwap();
	std::cout<<"coppied memory"<<std::endl;
	stbi_image_free(image_data);
	Tensor t_img = img_a.transpose(1,0);
	std::cout << "permuted"<<std::endl;
	if(dt == DType::uint8)
		return t_img.contiguous();
	std::cout<<"returning"<<std::endl;
	return t_img.to_dtype(dt);
}

}
}
