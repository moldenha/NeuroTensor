#include "numpy.h"
#include <iostream>
#include <sstream>
#include "exceptions.hpp"

namespace nt {
namespace functional {

void parse_shape(std::string_view input, std::vector<int64_t>& shape) {

    // Remove parentheses
    input.remove_prefix(1);
    input.remove_suffix(1);
    std::string inp(input);
    std::stringstream ss(inp); // Convert string_view to string for stringstream
    std::string token;

    while (std::getline(ss, token, ',')) {
        shape.push_back(std::stoll(token)); // Convert string to int64_t
    }

}

int parse_header(std::string header, DType& dt, std::vector<int64_t>& vec){
	std::string_view v_header(header);
	if(!(header[0] == '\x93' && v_header.substr(1, 5) == "NUMPY")){
		std::cerr << "Not a valid .npy file" << std::endl;
		return -1;
	}

	std::size_t pos = v_header.find("{");
	std::size_t end_pos = v_header.find("}");
	if(pos == std::string_view::npos || end_pos == std::string_view::npos){
		std::cerr << "Invalid header bounds" << std::endl;
		return -1;
	}

	v_header = v_header.substr(pos, end_pos - pos + 1);
	//extract dtype
	std::size_t dtype_pos = v_header.find("'descr': '");
	if(dtype_pos != std::string_view::npos){
		std::string_view dtype_str = v_header.substr(dtype_pos+10);
		dtype_str = dtype_str.substr(0, dtype_str.find("'"));
		if (dtype_str == "|u1") {
            dt = DType::uint8;
        } 
        else if (dtype_str == "|i1") {
            dt = DType::int8;
        }
        else if (dtype_str == "<i2") {
            dt = DType::int16;
        }
        else if (dtype_str == "<u2") {
            dt = DType::uint16;
        }
        else if (dtype_str == "<i4") {
            dt = DType::int32;
        }
        else if (dtype_str == "<u4") {
            dt = DType::uint32;
        }
        else if (dtype_str == "<i8") {
            dt = DType::int64;
        }
        else if (dtype_str == "<f4") {
            dt = DType::Float32;
        }
        else if (dtype_str == "<f8") {
            dt = DType::Float64;
        }
#ifdef _HALF_FLOAT_SUPPORT_
        else if (dtype_str == "<f2") {
            dt = DType::Float16;
        }
        else if (dtype_str == "<c4") {
            dt = DType::Complex32;
        }
#endif
#ifdef _128_FLOAT_SUPPORT_
        else if (dtype_str == "<f16") {
            dt = DType::Float128;
        }
#endif
#ifdef __SIZEOF_INT128__
        else if (dtype_str == "<i16") {
            dt = DType::int128;
        }
        else if (dtype_str == "<u16") {
            dt = DType::uint128;
        }
#endif
        else if (dtype_str == "<c8") {
            dt = DType::Complex64;
        }
        else if (dtype_str == "<c16") {
            dt = DType::Complex128;
        }
        else if (dtype_str == "|b1") {
            dt = DType::Bool;
        }
        else if(dtype_str == "tensor"){
            dt = DType::TensorObj;
        }
        else {
            std::cerr << "Unknown DType: " << dtype_str << std::endl;
            return -1;
        }
	}
    utils::throw_exception(dt != DType::TensorObj, "Currently tensors of tensors cannot be loaded or saved");
	std::size_t shape_pos = v_header.find("'shape': (");
	if(shape_pos != std::string_view::npos){
		std::string_view shape_str = v_header.substr(shape_pos+9);
		shape_str = shape_str.substr(0, shape_str.find(')')+1);
		parse_shape(shape_str, vec);
	}else{
		std::cerr << "Unable to parse shape" << std::endl;
		return -1;
	}
	return 0;
}


Tensor from_numpy(std::string filename){
	std::ifstream file(filename.c_str());
	if(!file.is_open()){
		std::cerr << "Error opening file!" << std::endl;
		return Tensor();
	}
	std::string line;
	if(!std::getline(file, line)){
		std::cerr << "Error getting line" << std::endl;
		file.close();
		return Tensor();
	}
	std::vector<int64_t> t_shape;
	DType t_dt;
	if(parse_header(std::move(line), t_dt, t_shape) != 0){
		file.close();
		return Tensor();
	}
	Tensor output(SizeRef(std::move(t_shape)), t_dt);
	void* data = output.data_ptr();

	// Move to the data position (after header)
	std::streampos header_end = file.tellg(); // Get current position
	file.seekg(0, std::ios::end); // Go to the end of the file
	std::streamsize file_size = file.tellg(); // Get the file size
	std::streamsize header_size = file_size - header_end; // Calculate header size
	file.seekg(header_end, std::ios::beg); // Go back to the end of the header
	file.read(static_cast<char*>(data), output.numel() * DTypeFuncs::size_of_dtype(t_dt));
	

	file.close();
	return std::move(output);
}
 
void to_numpy(const Tensor &tensor, std::string filename){
    _NT_FUNCTIONAL_ALWAYS_CHECK_(tensor);
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    // NumPy magic string and version
    file.write("\x93NUMPY", 6);
    uint8_t major_version = 1;
    uint8_t minor_version = 0;
    file.write(reinterpret_cast<char*>(&major_version), 1);
    file.write(reinterpret_cast<char*>(&minor_version), 1);

    // Tensor dtype conversion to NumPy dtype
    std::string dtype_str;
    utils::throw_exception(tensor.dtype != DType::TensorObj, "Currently tensors of tensors cannot be loaded or saved");
    switch (tensor.dtype) {
        case DType::Float32: dtype_str = "<f4"; break;
        case DType::Float64: dtype_str = "<f8"; break;
#ifdef _HALF_FLOAT_SUPPORT_
        case DType::Float16: dtype_str = "<f2"; break;
        case DType::Complex32: dtype_str = "<c4"; break;
#endif
#ifdef _128_FLOAT_SUPPORT_
        case DType::Float128: dtype_str = "<f16"; break;
#endif
#ifdef __SIZEOF_INT128__
        case DType::int128: dtype_str = "<i16"; break;
        case DType::uint128: dtype_str = "<u16"; break;
#endif
        case DType::Complex64: dtype_str = "<c8"; break;
        case DType::Complex128: dtype_str = "<c16"; break;
        case DType::uint8: dtype_str = "|u1"; break;
        case DType::int8: dtype_str = "|i1"; break;
        case DType::int16: dtype_str = "<i2"; break;
        case DType::uint16: dtype_str = "<u2"; break;
        case DType::int32: dtype_str = "<i4"; break;
        case DType::uint32: dtype_str = "<u4"; break;
        case DType::int64: dtype_str = "<i8"; break;
        case DType::Bool: dtype_str = "|b1"; break;
        case DType::TensorObj: dtype_str = "tensor"; break;
        default: dtype_str = "unknown"; break;
    }

    // Generate shape string
    std::string shape_str = "(";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        shape_str += std::to_string(tensor.shape()[i]);
        if (i + 1 < tensor.shape().size()) shape_str += ", ";
    }
    shape_str += (tensor.shape().size() == 1) ? ",)" : ")"; // Ensure correct formatting

    // Construct header dictionary
    std::string header = "{'descr': '" + dtype_str + "', 'fortran_order': False, 'shape': " + shape_str + ", }";
    // Align header to 64-byte boundary with padding spaces and newline
    int header_len = header.size() + 1;
    int padding = 64 - ((10 + header_len) % 64);
    header.append(padding, ' ');
    header += '\n';

    // Write header length
    uint16_t header_size = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    // Write header content
    file.write(header.c_str(), header.size());

    // Write tensor data
    Tensor cloned = tensor.clone();
    file.write(static_cast<const char*>(cloned.data_ptr()), cloned.numel() * DTypeFuncs::size_of_dtype(cloned.dtype));

    file.close();
    return;
}

} // namespace functional
} // namespace nt

