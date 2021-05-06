#include <iostream>
#include <iomanip>
#include <string>
#include "T_EvolutionGraphNN.h"

using namespace std;

template <class T>
void test(EvolutionGNN<T>& gnn, string printout, int iterations = 10) {
	cout << printout << endl;
	
	for(int i = 0; i < iterations; ++i){
		gnn.run();
		gnn.flipBuffer();
		cout << setw(7) << gnn.getOutput(0);
	}
}

int main(){

	cout << "Running test..." << endl;
	
	cout << "Constructing a GNN..." << endl;
	
	//The GNN will have 2 input and 1 output
	EvolutionGNN<float> gnn1(2, 1);
	
	//Add gate
	
	
	test(gnn1, "And gate with input [0, 0], expected output [0]");

	return 0;
}
