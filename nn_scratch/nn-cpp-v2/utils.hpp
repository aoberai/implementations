#include <assert.h>
#include <vector>
#include <string>

namespace utils {
	extern double END;
	extern double E;
	extern double EPSILON;
	std::string ToString(std::vector<double> a);
	// takes dot product of 2 identically size vectors
	double Dot(std::vector<double> a, std::vector<double> b);
	double Average(std::vector<double> a);
}

