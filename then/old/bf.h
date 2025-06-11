#ifndef _NT_BF_FILES_H_
#define _NT_BF_FILES_H_

#include <fstream>
#include <iostream>
#include <map>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cstring>
#include <deque>
#include <algorithm>


namespace nt{
namespace bf{
class BFInterpreter{
	std::deque<char> instruction;
	std::deque<char>::iterator instructionPointer, instructionPointerEnd;
	uint32_t dataPointer, bracket_counter;
	uint32_t data[65535];
	bool bracketf, bracketb;
	uint8_t *out_data;
	void handle_bracketf();
	void handle_bracketb();
	void increment(uint32_t* ptr, uint32_t max);
	void decrement(uint32_t* ptr, uint32_t max);
	public:
		BFInterpreter(uint8_t* outp);
		void handle_current();
		void interpret_character(const char c);
		BFInterpreter& operator<<(const char& c);
		BFInterpreter& operator<<(const char* c);
		BFInterpreter& operator<<(const std::string& c);
};

class BFFileGenerator{
	static std::map<char, uint8_t> keys;
	uint8_t count, current;
	std::ofstream& outfile;
	void write_bits();
	public:
		BFFileGenerator(std::ofstream& of);
		~BFFileGenerator();
		void add_character(const char& c);
		BFFileGenerator& operator<<(const char c);
		BFFileGenerator& operator<<(const char* c);
		BFFileGenerator& operator<<(const std::string& c);
};

class BFFileReader{
	static std::map<uint8_t, char> keys;
	public:
		BFFileReader();
		void read(std::ifstream& in, BFInterpreter& os);
		void read_till(std::ifstream& in, BFInterpreter& os, char t);

};

class BFGenerator{
	static std::array<std::array<std::string, 256>, 256> G;
	uint8_t lastc;
	public:
		BFGenerator();
		void generate(const uint8_t& c, std::ofstream& outfile);
		void generate(const uint8_t* begin, const uint8_t* end, std::ofstream& outfile, char delim);
		void generate(const uint8_t* begin, const uint8_t* end, std::ofstream& outfile);
		void generate(std::string::const_iterator begin, std::string::const_iterator end, std::ofstream& outfile);
		void generate(std::string::const_iterator begin, std::string::const_iterator end, std::ofstream& outfile, const char delim);
		void generate(const uint8_t& c, BFFileGenerator& outfile);
		void generate(const uint8_t* begin, const uint8_t* end, BFFileGenerator& outfile, const char delim);
		void generate(const uint8_t* begin, const uint8_t* end, BFFileGenerator& outfile);
		void generate(std::string::const_iterator begin, std::string::const_iterator end, BFFileGenerator& outfile);
		void generate(std::string::const_iterator begin, std::string::const_iterator end, BFFileGenerator& outfile, const char delim);
};


}
}

#endif //_NT_BF_FILES_H_
