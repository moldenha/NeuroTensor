#include "bf.h"

namespace nt{
namespace bf{
void BFInterpreter::handle_bracketf(){
	if(*instructionPointer == '\0'){
		throw std::logic_error("Unmatched [ -> ] bracket");
	}
	if(*instructionPointer == ']' && bracket_counter == 0){
		bracketf = false;
		return;
	}
	else if(*instructionPointer == ']'){
		bracket_counter--;
	}
	else if(*instructionPointer == '['){
		bracket_counter++;
	}
	++instructionPointer;
}

void BFInterpreter::handle_bracketb(){
	if(*instructionPointer == '[' && bracket_counter == 0){
		bracketb = false;
		return;
	}
	else if(*instructionPointer == '['){
		bracket_counter--;
	}
	else if(*instructionPointer == ']'){
		bracket_counter++;
	}
	if(instructionPointer == instruction.begin()){
		throw std::logic_error("Unmatched ] -> [ bracket");
		return;
	}
	instructionPointer = std::prev(instructionPointer);
}

void BFInterpreter::increment(uint32_t* ptr, uint32_t max){
	if(*ptr == max)
		*ptr = 0;
	else{
		++(*ptr);
	}
}

void BFInterpreter::decrement(uint32_t* ptr, uint32_t max){
	if(*ptr == 0)
		*ptr = max;
	else{
		--(*ptr);
	}		
}

BFInterpreter::BFInterpreter(uint8_t* outp)
	:dataPointer(0),
	instruction(65535),
	bracket_counter(0),
	bracketf(false),
	bracketb(false),
	out_data(outp)
{
	std::fill(instruction.begin(), instruction.end(), '\0');
	instructionPointer = instruction.begin();
	instructionPointerEnd = instruction.end();
	std::fill(&data[0], &data[65535], 0);
}


void BFInterpreter::handle_current(){
	while(*instructionPointer != '\0'){
		if(bracketf){
			handle_bracketf();
			continue;
		}
		if(bracketb){
			while(bracketb){handle_bracketb();}
			continue;
		}
		switch(*instructionPointer){
			case '>':
				increment(&dataPointer, 65535);
				break;
			case '<':
				decrement(&dataPointer, 65535);
				break;
			case '+':
				increment(&data[dataPointer], 255);
				break;
			case '-':
				increment(&data[dataPointer], 255);
				break;
			case '.':
				*out_data = data[dataPointer];
				++out_data;
				break;
			case ',':
				break;
			case '[':
				if(data[dataPointer] == 0){
					++instructionPointer;
					bracketf = true;
				}
				break;
			case ']':
				if(data[dataPointer] != 0){
					instructionPointer = std::prev(instructionPointer);
					bracketb = true;
				}
				break;
			default:
				break;
		}
		if(!bracketb && !bracketf)
			++instructionPointer;
	}
}

void BFInterpreter::interpret_character(const char c){
	if(instructionPointer == instructionPointerEnd){
		instruction.erase(instruction.begin(), instruction.begin() + 32767);
		instructionPointer = instruction.begin() + 32767;
		instruction.insert(instructionPointer, 32768, '\0');
		instructionPointer = instruction.begin() + 32767;
		instructionPointerEnd = instruction.end();
	}
	*instructionPointer = c;
	handle_current();

}

BFInterpreter& BFInterpreter::operator<<(const char& c){interpret_character(c);return *this;}
BFInterpreter& BFInterpreter::operator<<(const char* c){
	uint32_t size = strlen(c);
	for(uint32_t i = 0; i < size; ++i)
		interpret_character(c[i]);
	return *this;
}

BFInterpreter& BFInterpreter::operator<<(const std::string& c){
	auto beg = c.cbegin();
	auto end = c.cend();
	for(;beg != end; ++beg)
		interpret_character(*beg);
	return *this;
}

void BFFileGenerator::write_bits(){
	if(count == 0)
		return;
	if(count == 1)
		current = current << 4;
	outfile << (unsigned char)(current);
	count = 0;
	current = 0;
}

BFFileGenerator::BFFileGenerator(std::ofstream& of)
	:outfile(of), count(0), current(0)
{}

BFFileGenerator::~BFFileGenerator(){write_bits();}

void BFFileGenerator::add_character(const char& c){
	current = (count == 0) ? keys[c] : (current << 4) + keys[c];
	++count;
	if(count == 2) write_bits();
}

BFFileGenerator& BFFileGenerator::operator<<(const char c){add_character(c);return *this;}

BFFileGenerator& BFFileGenerator::operator<<(const char* c){
	uint32_t size = strlen(c);
	for(uint32_t i = 0; i < size; ++i)
		add_character(c[i]);
	return *this;
}


BFFileGenerator& BFFileGenerator::operator<<(const std::string& c){
	auto beg = c.cbegin();
	auto end = c.cend();
	for(;beg != end; ++beg)
		add_character(*beg);
	return *this;
}

void BFFileReader::read(std::ifstream& in, BFInterpreter& os){
	if(in.is_open()){
		while(in.good()){
			int8_t num = in.get();
			if(num == -1){
				break;
			}
			uint8_t casted = (uint8_t)num;
			uint8_t high_bits = casted >> 4;
			uint8_t low_bits = casted&15;
			os << keys[high_bits];
			if(low_bits != 0)
				os << keys[low_bits];
			
		}
	}
	if(!in.eof() && in.fail()){
		os << "error reading input";
		return;
	}
	in.close();
}

void BFFileReader::read_till(std::ifstream& in, BFInterpreter& os, char t){
	if(in.is_open()){
		while(in.good()){
			int8_t num = in.get();
			if(num == -1){
				break;
			}
			uint8_t casted = (uint8_t)num;
			uint8_t high_bits = casted >> 4;
			uint8_t low_bits = casted&15;
			if(keys[high_bits] == t)
				break;
			os << keys[high_bits];
			if(low_bits != 0){
				if(keys[low_bits] == t) break;
				os << keys[low_bits];
			}
			
		}
	}
	if(!in.eof() && in.fail()){
		os << "error reading input";
		return;
	}
	in.close();
}
BFFileReader::BFFileReader(){}

void BFGenerator::generate(const uint8_t& c, std::ofstream& outfile){
	std::string& a = G[lastc][c];
	std::string& b = G[0][c];
	if(a.size() <= b.size()) outfile << a << ".";
	else outfile << ">" << b << ".";
	lastc = c;
}
void BFGenerator::generate(const uint8_t* begin, const uint8_t* end, std::ofstream& outfile, char delim){
	for(;(begin+1) != end; ++begin){
		generate(*begin, outfile);
		outfile << delim;
	}
	++begin;
	generate(*begin, outfile);
}
void BFGenerator::generate(const uint8_t* begin, const uint8_t* end, std::ofstream& outfile){
	for(;begin != end;++begin)
		generate(*begin, outfile);
}
void BFGenerator::generate(std::string::const_iterator begin, std::string::const_iterator end, std::ofstream& outfile){
	for(;begin != end; ++begin)
		generate(*begin, outfile);
}

void BFGenerator::generate(std::string::const_iterator begin, std::string::const_iterator end, std::ofstream& outfile, const char delim){
	for(;begin != std::prev(end); ++begin){
		generate(*begin, outfile);
		outfile << delim;
	}
	++begin;
	generate(*begin, outfile);
}
void BFGenerator::generate(const uint8_t& c, BFFileGenerator& outfile){
	const std::string& a = G[lastc][c];
	const std::string& b = G[0][c];
	if(a.size() <= b.size()) outfile << a << '.';
	else outfile << '>' << b << '.';
	lastc = c;
}
void BFGenerator::generate(const uint8_t* begin, const uint8_t* end, BFFileGenerator& outfile, const char delim){
	for(;(begin+1) != end; ++begin){
		generate(*begin, outfile);
		outfile << delim;
	}
	++begin;
	generate(*begin, outfile);
}
void BFGenerator::generate(const uint8_t* begin, const uint8_t* end, BFFileGenerator& outfile){
	for(;begin != end;++begin)
		generate(*begin, outfile);
}
void BFGenerator::generate(std::string::const_iterator begin, std::string::const_iterator end, BFFileGenerator& outfile){
	for(;begin != end; ++begin)
		generate(*begin, outfile);
}

void BFGenerator::generate(std::string::const_iterator begin, std::string::const_iterator end, BFFileGenerator& outfile, const char delim){
	for(;begin != std::prev(end); ++begin){
		generate(*begin, outfile);
		outfile << delim;
	}
	++begin;
	generate(*begin, outfile);
}

std::map<char, uint8_t> BFFileGenerator::keys = std::map<char, uint8_t>({{'>',1},{'<',2},{'+',3},{'-',4},{'[',5},{']',6},{',',7},{'.',8},{'{',9},{'}',10}});

std::map<uint8_t, char> BFFileReader::keys = std::map<uint8_t, char>({{1,'>'},{2,'<'},{3,'+'},{4,'-'},{5,'['},{6,']'},{7,','},{8,'.'},{9,'{'},{10,'}'}}); 

/* std::array<std::array<std::string, 256>, 256> BFGenerator::G = []{ */
/* 	std::array<std::array<std::string, 256>, 256> values; */	
/* 	for(int_fast32_t x = 0; x < 256; ++x){ */
/* 		for(int_fast32_t y = 0; y < 256; ++y){ */
/* 			int_fast32_t delta = y - x; */
/* 			if(delta > 128) delta -= 256; */
/* 			if(delta < 128) delta += 256; */
/* 			if(delta >= 0) values[x][y] = std::string((std::size_t)delta, '+'); */
/* 			else values[x][y] = std::string((-delta), '-'); */
/* 		} */
/* 	} */
	
/* 	bool iter = true; */
/* 	while(iter){ */
/* 		iter = false; */
/* 		for(int_fast32_t x = 0; x < 256; ++x){ */
/* 			for(int_fast32_t n = 1; n < 40; ++n){ */
/* 				for(int_fast32_t d = 1; d < 40; ++d){ */
/* 					int_fast32_t j = x; */
/* 					int_fast32_t y = 0; */
/* 					for (int i = 0; i < 256; ++i){ */
/* 					    if (j == 0) break; */
/* 					    j = (j - d + 256) & 255; */
/* 					    y = (y + n) & 255; */
/* 					} */
/* 					if(j == 0){ */
/* 						if((5+d+n) < values[x][y].size()){ */
/* 							values[x][y] = "[" + std::string(d, '-') + ">"+ std::string(n, '+') + "<]>"; */
/* 							iter = true; */
/* 						} */
/* 					} */
/* 					j = x; */
/* 					y = 0; */
/* 					for (int i = 0; i < 256; ++i) { */
/* 					    if (j == 0) break; */
/* 					    j = (j + d) & 255; */
/* 					    y = (y - n + 256) & 255; */
/* 					} */
/* 					if (j == 0) { */
/* 					    if ((5+d+n) < values[x][y].size()) { */
/* 						values[x][y] = "[" + std::string(d, '+') + ">" + std::string(n, '-') + "<]>"; */
/* 						iter = true; */
/* 					    } */
/* 					} */
/* 				} */
/* 			} */
/* 		} */
/* 		// combine number schemes */                                                               
/* 		for (int x = 0; x < 256; ++x) { */
/* 			for (int y = 0; y < 256; ++y) { */
/* 				for (int z = 0; z < 256; ++z) { */
/* 					if (values[x][z].size() + values[z][y].size() < values[x][y].size()) { */
/* 						values[x][y] = values[x][z] + values[z][y]; */
/* 						iter = true; */
/* 					} */
/* 				} */
/* 			} */
/* 		} */
/* 	} */
/* 	return values; */
/* }(); */


}
}
