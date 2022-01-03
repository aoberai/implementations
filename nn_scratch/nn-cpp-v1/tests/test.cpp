#include <iostream>
#include<vector>
#include <random>
using namespace std;


void test(initializer_list<double> inputs) {
	cout << sizeof(inputs)/sizeof(double);
}
int main() {
	test({1, 2, 3});
	// random_device rd;
  //
	// mt19937 e2(rd());
	// normal_distribution<> dist(0.5, 0.15);
	// cout <<dist(e2);
	// vector<double> temp(5, 0);
	// for (auto &v : temp) {
	// 	v = dist(e2);
	// }
	// for (auto i : temp) {
	// 		cout << i << endl;
	// }

	// for (auto i : vector<double>(5, dist(e2))) {
		// rmal_distribution<> dist(70, 10);
		// cout << i << endl;
	// }
	// cout << to_string(5) + "asdf asDF";
	// vector<int> a = {4, 2, 3};
	//
	//
	//
	// // declare the vector of type int
	// // vector<int> v;
	//
	//
	// // print those elements
	// for (auto it = a.begin(); it != a.end(); ++it){
	// 	std::cout << *it << std::endl;
	// }
	return 0;
}
