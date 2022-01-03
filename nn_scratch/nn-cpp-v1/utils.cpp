#include "utils.hpp"
#include <vector>
#include <iostream>

using namespace std;
double utils::END = -1;
double utils::E = 2.71828;
double utils::EPSILON = 0.0000001;
double utils::dot_product(std::vector<double> a, std::vector<double> b) {
	assert(a.size() == b.size());
	
	// // print a
	// cout << " Vector A ";
	// for (auto v : a) {
	// 	cout << v << ", ";
	// }
	// cout << endl;	
  //
	// // print b
	// cout << " Vector B ";
	// for (auto v : b) {
	// 	cout << v << ", ";
	// }
	// cout << endl;	

	double value = 0;
	for (int i = 0; i < a.size(); i++) {
		value += a[i] * b[i];
	
	}

	// cout << "Dot Product: " << value << endl << endl;

	return value;
}

double utils::average(std::vector<double> a) {
	double summation = 0;
	for (double value : a) {
		summation += value;
	}
	return summation/a.size();
}


