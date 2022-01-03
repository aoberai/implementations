#include "utils.hpp"

double utils::END = -1;
double utils::E = 2.71828;
double utils::EPSILON = 0.0000001;
double utils::Dot(std::vector<double> a, std::vector<double> b) {
		assert(a.size() == b.size());
		double value = 0;
		for (int i = 0; i < a.size(); i++) {
			value += a[i] * b[i];

		}
		return value;
}


double utils::Average(std::vector<double> a) {
	double summation = 0;
	for (double value : a) {
		summation += value;
	}
	return summation/a.size();
}

std::string utils::ToString(std::vector<double> a) {
	std::string output = "[";
	for (double value : a) {
		output += std::to_string(value);
		output += ", ";
	}
	output += "]";
	return output;
}



