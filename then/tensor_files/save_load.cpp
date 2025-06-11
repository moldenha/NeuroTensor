#include "save_load.h"
#include "numpy.h"

namespace nt{
namespace functional{
Tensor load(std::string fname){return from_numpy(fname);}
Tensor load(const char* fname){return from_numpy(std::string(fname));}
void save(Tensor t, std::string fname){to_numpy(t, fname);}
void save(const Tensor& t, const char* fname){to_numpy(t, std::string(fname));}
}
}
