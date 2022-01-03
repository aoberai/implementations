#include <assert.h>
#include <vector>

namespace utils {
	extern double END;
	extern double E;
	extern double EPSILON;
	// takes dot product of 2 identically size vectors
	double dot_product(std::vector<double> a, std::vector<double> b);
	double average(std::vector<double> a);
}

