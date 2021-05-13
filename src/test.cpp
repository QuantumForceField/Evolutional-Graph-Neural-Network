#include <iostream>
#include <iomanip>
#include <string>
#include "T_EvolutionGraphNN.h"

using namespace std;

template <class T>
void test(EvolutionGNN<T>& gnn, string printout, int iterations = 10) {
	cout << endl << printout << endl;
	
	for(int i = 0; i < iterations; ++i){
		gnn.run();
		gnn.flipBuffer();
		cout << setw(7) << gnn.getOutput(0);
	}
	cout << endl;
}

int main(){

	cout << "Running test..." << endl;
	
	//Testing NOT operator
	EvolutionGNN<float> notGate(1, 1);
	//Add connections
	notGate.addConnection(0, 1, -20);
	//Set input
	notGate.setInput(0, -1);//Test
	//Test
	test(notGate, "NOT GATE with input [-1], expected output [ 1]");
	
	//Set input
	notGate.setInput(0, 1);//Test
	//Test
	test(notGate, "NOT GATE with input [ 1], expected output [-1]");
	
	cout << endl;
	//Save NOT gate as dot
	notGate.saveDOT("notgate.dot");
	
	
	
	//Testing OR operation
	EvolutionGNN<float> orGate(2, 1);
	//Add nodes
	orGate.addNodes(3);
	//Add connections
	orGate.addConnection(0, 4, 20);
	orGate.addConnection(1, 5, 20);
	orGate.addConnection(3, 3, 20, 1, 1);
	orGate.addConnection(3, 4, 20);
	orGate.addConnection(3, 5, 20);
	orGate.addConnection(3, 2, -20);
	orGate.addConnection(4, 2, 40);
	orGate.addConnection(5, 2, 40);
	//Set input
	orGate.setInput(0, -1);
	orGate.setInput(1, -1);
	//Test
	test(orGate, "OR GATE with input [-1, -1], expected output [-1]");
	
	//Set input
	orGate.setInput(0, 1);
	orGate.setInput(1, -1);
	//Test
	test(orGate, "OR GATE with input [1,  -1], expected output [ 1]");
	
	//Set input
	orGate.setInput(0, -1);
	orGate.setInput(1, 1);
	//Test
	test(orGate, "OR GATE with input [-1,  1], expected output [ 1]");
	
	//Set input
	orGate.setInput(0, 1);
	orGate.setInput(1, 1);
	//Test
	test(orGate, "OR GATE with input [1,   1], expected output [ 1]");
	
	cout << endl;
	//Save OR gate as dot
	orGate.saveDOT("orgate.dot");
	
	
	
	//Testing OR operation
	EvolutionGNN<float> andGate(2, 1);
	//Add nodes
	andGate.addNodes(3);
	//Add connections
	andGate.addConnection(0, 2, 40);
	andGate.addConnection(1, 2, 40);
	andGate.addConnection(3, 3, 20, 1, 1);
	andGate.addConnection(3, 2, -60);
	//Set input
	andGate.setInput(0, -1);
	andGate.setInput(1, -1);
	//Test
	test(andGate, "AND GATE with input [-1, -1], expected output [-1]");
	
	//Set input
	andGate.setInput(0, 1);
	andGate.setInput(1, -1);
	//Test
	test(andGate, "AND GATE with input [1,  -1], expected output [-1]");
	
	//Set input
	andGate.setInput(0, -1);
	andGate.setInput(1, 1);
	//Test
	test(andGate, "AND GATE with input [-1,  1], expected output [-1]");
	
	//Set input
	andGate.setInput(0, 1);
	andGate.setInput(1, 1);
	//Test
	test(andGate, "AND GATE with input [1,   1], expected output [ 1]");
	
	cout << endl;
	//Save AND gate as dot
	andGate.saveDOT("andgate.dot");
	
	
	
	//Testing to save neural network
	cout << "Saving AND_GATE.TEvoGNN..." << endl;
	andGate.save("AND_GATE.TEvoGNN");
	cout << "Done." << endl;
	
	//Testing to load neural network
	EvolutionGNN<float> loaded;
	cout << "Loading from AND_GATE.TEvoGNN" << endl;
	loaded.load("AND_GATE.TEvoGNN");
	//Set input
	loaded.setInput(0, -1);
	loaded.setInput(1, -1);
	//Test
	test(loaded, "Loaded AND GATE with input [-1, -1], expected output [-1]");
	
	//Set input
	loaded.setInput(0, 1);
	loaded.setInput(1, -1);
	//Test
	test(loaded, "Loaded AND GATE with input [1,  -1], expected output [-1]");
	
	//Set input
	loaded.setInput(0, -1);
	loaded.setInput(1, 1);
	//Test
	test(loaded, "Loaded AND GATE with input [-1,  1], expected output [-1]");
	
	//Set input
	loaded.setInput(0, 1);
	loaded.setInput(1, 1);
	//Test
	test(loaded, "Loaded AND GATE with input [1,   1], expected output [ 1]");
	
	cout << endl;
	
	
	
	//Following section demostrate mutation, inheritance and saving as DOT
	
	//Generate a random network
	EvolutionGNN<float> a;
	//Initialize with 5 inputs and 5 outputs
	a.initialize(5, 5);
	//Add 10 hidden nodes
	a.addNodes(10);
	//Mutate for 20 times
	for(int i = 0; i < 20; ++i)
		a.mutate(0.9, 0.05, 0.0);
	//Save DOT
	a.saveDOT("aNetwork.dot");
	
	
	//Generate a random network
	EvolutionGNN<float> b;
	//Initialize with 5 inputs and 5 outputs
	b.initialize(5, 5);
	//Add 15 hidden nodes
	b.addNodes(15);
	//Mutate for 10 times
	for(int i = 0; i < 10; ++i)
		b.mutate(0.9, 0.05, 0.0);
	//Save DOT
	b.saveDOT("bNetwork.dot");
	
	//Create a network that inheritant from a and b
	EvolutionGNN<float> c;
	//Inherit from a and b
	c.inherit(a, b, 0.9, 0.9);
	//Save DOT
	c.saveDOT("cNetwork.dot");
	
	return 0;
}
