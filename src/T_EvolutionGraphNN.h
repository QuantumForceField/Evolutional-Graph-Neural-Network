/*
* Date-format:		DD-MM-YYYY
* Creation-date:	09-04-2021
* Last-updated:		07-05-2021
*
* File-name:	T_EvolutionGraphNN.h
* Version:		0.0.3
* Author:		QuantumForceField
* Describtion:	T_EvolutionGraphNN.h contains implementation of graph neural network
*				that can be trained by evolutional process (e.g. genetic algorithms)
*/

#pragma once
#ifndef T_EVOLUTIONGRAPHNN_H
#define T_EVOLUTIONGRAPHNN_H

#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <queue>
#include <exception>
#include <math.h>
#include <thread>
#include <string>
#include <cstring>
#include <fstream>
#include <random>

using namespace std;

#define DEBUG


// Connection manages a connection
// The connection is directed from inNode to outNode
// Input node's id is stored as inNodeId
// Output node's id is stored as outNodeId
// Each connection is only a single connection
template <class T>
class Connection {
protected:
	//Indicating with buffer to use
	//If useABuffer == true, all write action will be performed on ABuffer
	//and all read action will be performed on BBuffer
	bool useABuffer;

	T weight;	//Connection weight

	// Two buffers which will be switching for read/write actions
	T ABuffer;
	T BBuffer;

	int inNodeId;	//Id of input node
	int outNodeId;	//Id of output node

public:

	//Initializer with default constructor
	Connection();

	//Initializer with known inNode and outNode
	//ABuffer, BBuffer, useABuffer can be passed optionally
	Connection(int inNodeId, int outNodeId, T weight = T(1), T ABuffer = T(0), T BBuffer = T(0), bool useABuffer = false);

	//Swap read and write buffer
	//Should be called after each timestep during simulation
	void flipBuffer();

	//Write operator overload
	void operator=(T val);

	//Get useABuffer
	bool getBufferState();

	//Set useABuffer
	void setBufferState(bool bufferState);

	//Get value
	T get();

	//Get ABuffer
	T getABuffer();

	//Get BBuffer
	T getBBuffer();

	//Get weight
	T getWeight();

	//Set weight
	void setWeight(T val);

	//Get inNodeId
	int getInNodeId();

	//Set inNodeId
	void setInNodeId(int inNodeId);

	//Set inNodeId to -1
	void removeInNodeId();

	//Get outNodeId
	int getOutNodeId();

	//Set outNodeId
	void setOutNodeId(int outNodeId);

	//Set inNodeId to -1
	void removeOutNodeId();

	//Check if this Connection is not connected with either inNode or outNode
	bool disconnected();

	//Used when writing to file
	void writeToFile(fstream& out);

	//For << overload
	friend ostream;
};

//Show details of the Connection
template <class T>
ostream& operator<<(ostream& o, Connection<T>& con) {
	o << "InNode: " << con.getInNodeId() << ", OutNode: " << con.getOutNodeId() << ", Weight = " << con.getWeight();
	o << ", A = " << con.getABuffer() << ", B = " << con.getBBuffer() << ", S = " << con.getBufferState();

	return o;
}

// GraphNode represents a single neuron in the GNN
// GraphNode manages two lists of Connections: inCon, outCon
// GraphNode also has a unique id
template <class T>
class GraphNode {
protected:

	int id;	//Unique Id
	vector<shared_ptr<Connection<T>>> inCon;	//Incoming connections
	vector<shared_ptr<Connection<T>>> outCon;	//Outgoing connections

public:

	//Initializer allowing to set id
	//inCon and outCon are initialized as empty
	GraphNode(int id = 0);

	//Set Id
	void setId(int id);

	//Get id
	int getId();

	//Add incoming Connection ptr
	void addInCon(shared_ptr<Connection<T>> in);

	//Remove incoming Connection ptr
	void removeInCon(shared_ptr<Connection<T>> in);

	//Add out-going Connection ptr
	void addOutCon(shared_ptr<Connection<T>> out);

	//Remove out-going Connection ptr
	void removeOutCon(shared_ptr<Connection<T>> out);

	//Flip all outgoing connection buffers
	void flipBuffer();

	//Run the neuron once
	void run();

	//Remove disconnected connections
	//Call this once in a while to clean up useless connections
	void removeDisconnectedConnections();

	//Get inCon vector
	vector<shared_ptr<Connection<T>>>& getInCon();

	//Get outCon vector
	vector<shared_ptr<Connection<T>>>& getOutCon();
};

//Input GraphNode is a special type of GraphNode
template <class T>
class InputGraphNode :public GraphNode<T> {
protected:
	T input;	//Used to remember input
public:

	//Constructor with optional id
	InputGraphNode(int id = 0) :GraphNode<T>(id) {
		input = T(0);
	}

	//Assignment opeartion, Assign input value
	T operator=(T val);

	//Run, assigning input to all outCon
	void run();
};

//Output GraphNode is a special type of GraphNode
template <class T>
class OutputGraphNode :public GraphNode<T> {
protected:
	T output;	//Used to remember output
public:

	//Constructor with optional id
	OutputGraphNode(int id = 0) :GraphNode<T>(id) {
		output = T(0);
	}

	//Return current computed output
	T get();

	//Run, calculate activation(sum of input)
	void run();
};

// EvolutionGNN is the entire envolutional graph neural network
// It manages a list of GraphNode using a hashtable hashing by it's id
// It manages a list of connections used in the graph neural network
// It also manages a list of GraphNodes that acted as input
// It also manages a list of GraphNodes that acted as output
template <class T>
class EvolutionGNN {
protected:

	//All input nodes, fixed, and will not be changed
	vector<InputGraphNode<T>> inputNodes;

	//All output nodes, fixed, and will not be changed
	vector<OutputGraphNode<T>> outputNodes;

	//All available nodes inside the graph excluding inputNodes and outputNodes
	//Index of hidden nodes will starts from |inputNodes.size() + outputNodes.size()|
	unordered_map<int, GraphNode<T>> graphNodes;
	vector<shared_ptr<Connection<T>>> con;

	//Total number of nodes
	int nodeCount;

	//Number of thread used to run
	int threadCount;

public:

	//Construct empty EvolutionGNN
	EvolutionGNN(int threadCount = -1);

	//Constructor with known input and output size
	EvolutionGNN(int inputCount, int outputCount, int threadCount = -1);

	//Constructor by inheritance from parents
	EvolutionGNN(EvolutionGNN<T>& parentA, EvolutionGNN<T>& parentB, double AConRate = 0.7, double BConRate = 0.3, bool inheritMemory = false);

	//Just like Constructor, initialize with known input and output size
	void initialize(int inputCount, int outputCount, int threadCount = -1);

	//Get the number of input
	int getInputSize();

	//Get the number of hidden nodes
	int getHiddenSize();

	//Get the number of output
	int getOutputSize();

	//Get the number of connections
	int getConnectionSize();

	//Clean up everything
	void cleanUp();

	//Set input to each inputNode
	void setInput(int index, T val);

	//Flip buffer for next run
	void flipBuffer();

	//Flip buffer in a multi-threaded way, should be called by flipBuffer()
	void thread_flipBuffer(int startId, int endId, int dummy = 0);

	//thread based runer
	//void t_run(int startNodeId, int endNodeId);

	//Run the whole neural network
	void run();

	//Run in a multi-threaded way, should be called by run()
	void thread_run(int startId, int endId, int dummy = 0);

	//Determine number of thread to run
	int determineNumberOfThread();

	//Task arranger function, set protion of tasks to threads
	double taskArranger(double x);

	//Get output from each outputNode
	T getOutput(int index);

	//Add hidden neuron
	void addNodes(int count = 1);

	//Add connection between nodes
	// node1 -------> node2
	void addConnection(int node1, int node2, T weight = T(1), T ABuffer = T(0), T BBuffer = T(0), bool useABuffer = false);

	//Add connection between nodes with random input
	// node1 -------> node2
	//weight following 
	void addRandomConnection(int count = 1, unsigned int randomState = rand());

	//Remove useless connections that are not connected to any nodes
	void removeDisconnectedConnections();

	//Save to file
	void save(string filename = "./out.TEvoGNN");

	//Load from file
	bool load(string path = "./out.TEvoGNN");

	//Cross bread
	//Accept two parents and selectively inherit their structures
	//inputNodes = max(parentA.inputNodes.size(), parentB.inputNodes.size())
	//outputNodes = max(parentA.inputNodes.size(), parentB.inputNodes.size())
	//graphNodes = max(parentA.graphNodes.size(), parentB.graphNodes.size())
	//AConRate: Percentage of connections been selected from parentA
	//BConRate: Percentage of connections been selected from parentB
	//inheritMemory: Select weither value and buffer states will be passed
	void inherit(EvolutionGNN<T>& parentA, EvolutionGNN<T>& parentB, double AConRate = 0.7, double BConRate = 0.3, bool inheritMemory = false);

	//Mutate
	//Random action
	//void mutate(double newConRate, double newNodeRate, double ...)


	//Testing function
	void test(bool mt = false) {
		srand(42);
		cleanUp();
		if (mt)initialize(1, 1);
		else
			initialize(1, 1, 1);

		////Add extra 5 nodes
		//for (int i = 0; i < 5; ++i)
		//	addNodes();
		////Add connections
		//addConnection(0, 2, 30.0);
		//addConnection(2, 3, 30.0);
		//addConnection(3, 4, 30.0);
		//addConnection(4, 5, 30.0);
		//addConnection(5, 6, 30.0);
		//addConnection(6, 1, 1.0);
		////Loop back
		//addConnection(6, 2, 30.0);


		//Random initializations
		for (int i = 0; i < 1000; ++i)
			addNodes();
		for (int i = 0; i < 10000; ++i)
			addConnection(rand() % nodeCount, rand() % nodeCount, rand() % 2001 / 1000.0 - 1.0);


		//OR gate
		//for (int i = 0; i < 4; ++i)
		//	addNodes();
		////Add connections
		//addConnection(0, 2);
		//addConnection(0, 3);
		//addConnection(2, 4, 40.0);
		//addConnection(3, 4, 40.0);
		//addConnection(5, 5, 40.0, 1.0, 1.0);	//Bias connection, constantly 1.0
		//addConnection(5, 4, -20.0);				//Bias connection
		//addConnection(4, 1, 40.0);

		////AND gate
		//for (int i = 0; i < 6; ++i)
		//	addNodes();
		////Add connections
		//addConnection(0, 2);
		//addConnection(0, 3);
		//addConnection(2, 4, -30.0);
		//addConnection(3, 5, -30.0);
		//addConnection(4, 6, 30.0);
		//addConnection(5, 6, 30.0);
		//addConnection(6, 7, -30.0);
		//addConnection(7, 1, 1.0);

		removeDisconnectedConnections();

		//Set initial input
		setInput(0, 1);
		run();
		flipBuffer();

		for (int i = 0; i < 100; ++i) {
			//setInput(0, 0);
			run();
			flipBuffer();
			printf("%7.3f", getOutput(0));
			if (i % 15 == 0) cout << endl;
		}
		cout << endl << endl;

		cout << "Writing to file..." << endl;
		if (!mt) {
			save("./single_thread.TEvoGNN");
			load("./single_thread.TEvoGNN");
		}
		else {
			save("./multi_thread.TEvoGNN");
			load("./multi_thread.TEvoGNN");
		}
		cout << "Done!" << endl;

		cout << "Continue simulation after reloading from file" << endl;srand(42);
		removeDisconnectedConnections();

		for (int i = 0; i < 100; ++i) {
			setInput(0, 0);
			run();
			flipBuffer();
			printf("%7.3f", getOutput(0));
			if (i % 15 == 0) cout << endl;
		}
		cout << endl << endl << endl;;
	}


	friend ostream;
};

//Show info of a EvolutionGNN
template <class T>
ostream& operator<<(ostream& o, EvolutionGNN<T>& egnn) {
	o << "Evolution Graph Neural Network" << endl;
	o << "\tInput Nodes:\t" << egnn.getInputSize() << endl;
	o << "\tHidden Nodes:\t" << egnn.getHiddenSize() << endl;
	o << "\tOutput Nodes:\t" << egnn.getOutputSize() << endl;
	o << "\tConnections:\t" << egnn.getConnectionSize();
	return o;
}

/***********************************************/
// Function bodies

template <class T>
void EvolutionGNN<T>::inherit(EvolutionGNN<T>& parentA, EvolutionGNN<T>& parentB, double AConRate, double BConRate, bool inheritMemory) {

	//Check required number of nodes
	int inNodeCount = parentA.inputNodes.size();
	int outNodeCount = parentA.outputNodes.size();
	int hiddenNodeCount = parentA.graphNodes.size();
	if (parentB.inputNodes.size() > inNodeCount)inNodeCount = parentB.inputNodes.size();
	if (parentB.outputNodes.size() > outNodeCount)outNodeCount = parentB.outputNodes.size();
	if (parentB.graphNodes.size() > hiddenNodeCount)hiddenNodeCount = parentB.graphNodes.size();

	//Create all nodes
	initialize(inNodeCount, outNodeCount, threadCount);
	addNodes(hiddenNodeCount);

	//Selectively add connections from parents
	//Note that if we also add buffer related info(values, states) to the child,
	//"memory" will be passed to the child
	for (auto i : parentA.con)
		if ((rand() % 1000) / 1000.0 < AConRate)
			if (inheritMemory)
				addConnection(i->getInNodeId(), i->getOutNodeId(), i->getWeight(), i->getABuffer(), i->getBBuffer(), i->getBufferState());
			else
				addConnection(i->getInNodeId(), i->getOutNodeId(), i->getWeight());

	for (auto i : parentB.con)
		if ((rand() % 1000) / 1000.0 < BConRate)
			if (inheritMemory)
				addConnection(i->getInNodeId(), i->getOutNodeId(), i->getWeight(), i->getABuffer(), i->getBBuffer(), i->getBufferState());
			else
				addConnection(i->getInNodeId(), i->getOutNodeId(), i->getWeight());

}

template <class T>
bool EvolutionGNN<T>::load(string path) {

	fstream in;
	in.open(path, ios::in | ios::binary);

	//Check if file open succeed
	if (!in.is_open())return false;

	char str[32] = { 0 };
	in.read(str, 11);
	//cout << "Read: " << str << endl;
	//Check if keyword correct
	if (strcmp(str, "InputNodes=")) {
		//Error
		in.close();
		//cout << "Wrong keyword." << endl;
		return false;
	}

	//Get InputNodes
	int inputNodes;
	in >> inputNodes;

	//cout << "InputNodes=" << inputNodes << endl;
	if (inputNodes < 0) {
		//Error
		in.close();
		//cout << "Wrong inputNodes." << endl;
		return false;
	}


	in.read(str, 13);
	//cout << "Read: " << str << endl;//Check if keyword correct
	if (strcmp(str, "\nHiddenNodes=")) {
		//Error
		in.close();
		//cout << "Wrong keyword." << endl;
		return false;
	}

	//Get HiddenNodes
	int hiddenNodes;
	in >> hiddenNodes;

	//cout << "HiddenNodes=" << hiddenNodes << endl;
	if (hiddenNodes < 0) {
		//Error
		in.close();
		//cout << "Wrong hiddenNodes." << endl;
		return false;
	}


	in.read(str, 13);
	//cout << "Read: " << str << endl;//Check if keyword correct
	if (strcmp(str, "\nOutputNodes=")) {
		//Error
		in.close();
		//cout << "Wrong keyword." << endl;
		return false;
	}

	//Get outputNodes
	int outputNodes;
	in >> outputNodes;

	//cout << "OutputNodes=" << outputNodes << endl;
	if (outputNodes < 0) {
		//Error
		in.close();
		//cout << "Wrong outputNodes." << endl;
		return false;
	}


	in.read(str, 13);
	//cout << "Read: " << str << endl;//Check if keyword correct
	if (strcmp(str, "\nConnections=")) {
		//Error
		in.close();
		//cout << "Wrong keyword." << endl;
		return false;
	}

	//Get HiddenNodes
	int connections;
	in >> connections;

	//cout << "Connections=" << connections << endl;
	if (connections < 0) {
		//Error
		in.close();
		//cout << "Wrong connections." << endl;
		return false;
	}

	//Read extra '\n'
	in.read(str, 1);

	//Prepare for initialization
	cleanUp();

	//Initialize with given inputNodes and outputNodes
	initialize(inputNodes, outputNodes);

	//Add all nodes
	addNodes(hiddenNodes);

	//Run for each connection
	int inNode, outNode;
	T weight, ABuffer, BBuffer;
	bool useABuffer;
	try {
		//cout << "Entered" << endl;
		for (int i = 0; i < connections; ++i) {
			in.read(reinterpret_cast<char*>(&inNode), sizeof(int));
			in.read(reinterpret_cast<char*>(&outNode), sizeof(int));
			in.read(reinterpret_cast<char*>(&weight), sizeof(T));
			in.read(reinterpret_cast<char*>(&ABuffer), sizeof(T));
			in.read(reinterpret_cast<char*>(&BBuffer), sizeof(T));
			in.read(reinterpret_cast<char*>(&useABuffer), sizeof(bool));
			//cout << "I=" << inNode << "\tO=" << outNode << "\tw=" << weight << "\tA=" << ABuffer << "\tB=" << BBuffer << "\tS=" << useABuffer << endl;
			addConnection(inNode, outNode, weight, ABuffer, BBuffer, useABuffer);
		}
	}
	catch (fstream::failure e) {
		cerr << "Load operation failed." << endl;
		in.close();
		return false;
	}

	in.close();

	return true;
}

template <class T>
void EvolutionGNN<T>::save(string filename) {
	fstream output(filename, ios::out | ios::binary);

	//Write input nodes
	output << "InputNodes=" << inputNodes.size() << endl;

	//Write hidden nodes
	output << "HiddenNodes=" << graphNodes.size() << endl;

	//Write output nodes
	output << "OutputNodes=" << outputNodes.size() << endl;

	//Write all connections
	output << "Connections=" << con.size() << endl;
	for (shared_ptr<Connection<T>> ptr : con)
		ptr->writeToFile(output);

	output.close();
}

template <class T>
void EvolutionGNN<T>::removeDisconnectedConnections() {
	//Remove useless connectinos for each input Node
	for (int i = 0; i < inputNodes.size(); ++i)
		inputNodes[i].removeDisconnectedConnections();

	//Remove useless connectinos for each output Node
	for (int i = 0; i < outputNodes.size(); ++i)
		outputNodes[i].removeDisconnectedConnections();

	//Remove useless connections for each hidden node
	for (auto i = graphNodes.begin(); i != graphNodes.end(); ++i)
		i->second.removeDisconnectedConnections();

	//Remove disconnected connections
	vector<int> indexes;
	for (int i = 0; i < con.size(); ++i)
		if (con[i]->disconnected())indexes.push_back(i);

	for (int i = indexes.size() - 1; i >= 0; --i) {
		con.erase(con.begin() + indexes[i]);
		indexes.pop_back();
	}
}

template <class T>
void EvolutionGNN<T>::addRandomConnection(int count, unsigned int randomState) {
	default_random_engine generator(randomState);
	normal_distribution<double> distribution(1.0, 1.0);
	for (int i = 0; i < count; ++i)
		addConnection(rand() % nodeCount, rand() % nodeCount, rand() % 20000 / 1000.0 - 10.0);
}

template <class T>
void EvolutionGNN<T>::addConnection(int node1, int node2, T weight, T ABuffer, T BBuffer, bool useABuffer) {

	//Create Connection
	shared_ptr<Connection<T>> ptr = make_shared<Connection<T>>(node1, node2, weight, ABuffer, BBuffer, useABuffer);

	//Added to Connections
	con.push_back(ptr);

	//Added as outCon to node1
	if (node1 < inputNodes.size())
		inputNodes[node1].addOutCon(ptr);
	else
		if (node1 < inputNodes.size() + outputNodes.size())
			outputNodes[node1 - inputNodes.size()].addOutCon(ptr);
		else
			graphNodes[node1].addOutCon(ptr);

	//Added as inCon to node2
	if (node2 < inputNodes.size())
		inputNodes[node2].addInCon(ptr);
	else
		if (node2 < inputNodes.size() + outputNodes.size())
			outputNodes[node2 - inputNodes.size()].addInCon(ptr);
		else
			graphNodes[node2].addInCon(ptr);
}

template <class T>
void EvolutionGNN<T>::addNodes(int count) {
	for (int i = 0; i < count; ++i) {
		graphNodes.emplace(nodeCount, GraphNode<T>(nodeCount));
		++nodeCount;
	}
}

template <class T>
T EvolutionGNN<T>::getOutput(int index) {
	return outputNodes[index].get();
}

template <class T>
double EvolutionGNN<T>::taskArranger(double x) {
	//return pow(x, M_E);
	return x;
}

template <class T>
int EvolutionGNN<T>::determineNumberOfThread() {
	//Current method depends on number of connections
	int maxThread = threadCount;
	if (maxThread <= 0)maxThread = 1;

	int calculated = con.size() / 100000;
	if (calculated <= 0)calculated = 1;
	if (calculated > maxThread)calculated = maxThread;

	return calculated;
}

template <class T>
void EvolutionGNN<T>::thread_run(int startId, int endId, int dummy) {
	//cout << "Id = " << dummy << "  from " << startId << " to " << endId << endl;
	//cout << dummy << " Started." << endl;
	//Run inputNodes
	if (startId < inputNodes.size()) {
		int start = startId;
		int end = (endId < inputNodes.size() ? endId : inputNodes.size());
		for (int i = start; i < end; ++i)
			inputNodes[i].run();
	}

	//Run outputNodes
	if (startId < inputNodes.size() + outputNodes.size() && endId > inputNodes.size()) {
		int start = (startId < inputNodes.size() ? inputNodes.size() : startId) - inputNodes.size();
		int end = (endId > inputNodes.size() + outputNodes.size() ? inputNodes.size() + outputNodes.size() : endId) - inputNodes.size();
		for (int i = start; i < end; ++i)
			outputNodes[i].run();
	}

	//Run hiddenNodes
	if (endId > inputNodes.size() + outputNodes.size()) {
		int start = (startId < inputNodes.size() + outputNodes.size() ? inputNodes.size() + outputNodes.size() : startId) - inputNodes.size() - outputNodes.size();
		int end = endId - inputNodes.size() - outputNodes.size();
		auto s = graphNodes.begin();
		//Loop until s point to the correct one
		for (int i = 0; i < start; ++i)s++;
		int count = end - start;
		for (int i = 0; i < count; ++i) {
			s->second.run();
			s++;
		}
	}
	//cout << dummy << " Completed." << endl;
}

template <class T>
void EvolutionGNN<T>::run() {
	int numOfThread = determineNumberOfThread();
	if (numOfThread <= 1) {
		//Order doesn't matter

		//Run all input nodes
		for (int i = 0; i < inputNodes.size(); ++i)
			inputNodes[i].run();

		//Run all output nodes
		for (int i = 0; i < outputNodes.size(); ++i)
			outputNodes[i].run();

		//Run all hidden nodes
		for (auto i = this->graphNodes.begin(); i != this->graphNodes.end(); i++)
			i->second.run();
	}
	else {

		//Considering multi-threaded execution
		vector<thread> threadPool;
		for (int i = numOfThread - 1; i >= 0; --i)
			threadPool.push_back(thread(&EvolutionGNN<T>::thread_run, this, taskArranger(1.0 * i / numOfThread) * nodeCount, taskArranger(1.0 * (i + 1) / numOfThread) * nodeCount, i));

		//Wait for all thread to finish
		for (int i = 0; i < threadPool.size(); ++i)
			threadPool[i].join();
	}
}

template <class T>
void EvolutionGNN<T>::thread_flipBuffer(int startId, int endId, int dummy) {
	//cout << "Id = " << dummy << "  from " << startId << " to " << endId << endl;
	//cout << dummy << " Started." << endl;
	//Run inputNodes
	if (startId < inputNodes.size()) {
		int start = startId;
		int end = (endId < inputNodes.size() ? endId : inputNodes.size());
		for (int i = start; i < end; ++i)
			inputNodes[i].flipBuffer();
	}

	//Run outputNodes
	if (startId < inputNodes.size() + outputNodes.size() && endId > inputNodes.size()) {
		int start = (startId < inputNodes.size() ? inputNodes.size() : startId) - inputNodes.size();
		int end = (endId > inputNodes.size() + outputNodes.size() ? inputNodes.size() + outputNodes.size() : endId) - inputNodes.size();
		for (int i = start; i < end; ++i)
			outputNodes[i].flipBuffer();
	}

	//Run hiddenNodes
	if (endId > inputNodes.size() + outputNodes.size()) {
		int start = (startId < inputNodes.size() + outputNodes.size() ? inputNodes.size() + outputNodes.size() : startId) - inputNodes.size() - outputNodes.size();
		int end = endId - inputNodes.size() - outputNodes.size();
		auto s = graphNodes.begin();
		//Loop until s point to the correct one
		for (int i = 0; i < start; ++i)s++;
		int count = end - start;
		for (int i = 0; i < count; ++i) {
			s->second.flipBuffer();
			s++;
		}
	}
	//cout << dummy << " Completed." << endl;
}

template <class T>
void EvolutionGNN<T>::flipBuffer() {
	int numOfThread = determineNumberOfThread();
	if (numOfThread <= 1) {
		//Order doesn't matter

		//Flip input Nodes
		for (int i = 0; i < inputNodes.size(); ++i)
			inputNodes[i].flipBuffer();

		//Flip output Nodes
		for (int i = 0; i < outputNodes.size(); ++i)
			outputNodes[i].flipBuffer();

		//Flip hidden Nodes
		for (auto i = graphNodes.begin(); i != graphNodes.end(); i++)
			i->second.flipBuffer();
	}
	else {
		vector<thread> threadPool;
		for (int i = numOfThread - 1; i >= 0; --i)
			threadPool.push_back(thread(&EvolutionGNN<T>::thread_flipBuffer, this, taskArranger(1.0 * i / numOfThread) * nodeCount, taskArranger(1.0 * (i + 1) / numOfThread) * nodeCount, i));

		//Wait for all thread to finish
		for (int i = 0; i < threadPool.size(); ++i)
			threadPool[i].join();
	}
}

template <class T>
void EvolutionGNN<T>::setInput(int index, T val) {
	inputNodes[index] = val;
}

template <class T>
void EvolutionGNN<T>::cleanUp() {
	this->inputNodes.clear();
	this->outputNodes.clear();
	this->graphNodes.clear();
	this->con.clear();
}

template <class T>
int EvolutionGNN<T>::getConnectionSize() {
	return con.size();
}

template <class T>
int EvolutionGNN<T>::getOutputSize() {
	return outputNodes.size();
}

template <class T>
int EvolutionGNN<T>::getHiddenSize() {
	return graphNodes.size();
}

template <class T>
int EvolutionGNN<T>::getInputSize() {
	return inputNodes.size();
}

template <class T>
void EvolutionGNN<T>::initialize(int inputCount, int outputCount, int threadCount) {
	if (threadCount < 0)
		this->threadCount = thread::hardware_concurrency() - 1;
	else
		this->threadCount = threadCount;
	if (this->threadCount <= 0)
		this->threadCount = 1;

	//Remove everything in case not
	this->cleanUp();

	//Initialize it as initializer
	for (int i = 0; i < inputCount; ++i)
		this->inputNodes.push_back(InputGraphNode<T>(i));
	for (int i = 0; i < outputCount; ++i)
		this->outputNodes.push_back(OutputGraphNode<T>(i + inputCount));

	nodeCount = inputCount + outputCount;
}

template <class T>
EvolutionGNN<T>::EvolutionGNN(EvolutionGNN<T>& parentA, EvolutionGNN<T>& parentB, double AConRate, double BConRate, bool inheritMemory) {
	inherit(parentA, parentB, AConRate, BConRate, inheritMemory);
}

template <class T>
EvolutionGNN<T>::EvolutionGNN(int inputCount, int outputCount, int threadCount) {
	if (threadCount < 0)
		this->threadCount = thread::hardware_concurrency() - 1;
	else
		this->threadCount = threadCount;
	if (this->threadCount <= 0)
		this->threadCount = 1;

	for (int i = 0; i < inputCount; ++i)
		this->inputNodes.push_back(InputGraphNode<T>(i));
	for (int i = 0; i < outputCount; ++i)
		this->outputNodes.push_back(OutputGraphNode<T>(i + inputCount));

	nodeCount = inputCount + outputCount;
}

template <class T>
EvolutionGNN<T>::EvolutionGNN(int threadCount) {
	if (threadCount < 0)
		this->threadCount = thread::hardware_concurrency() - 1;
	else
		this->threadCount = threadCount;
	if (this->threadCount <= 0)
		this->threadCount = 1;
	nodeCount = 0;
}

template <class T>
void OutputGraphNode<T>::run() {
	T sum = T(0);
	for (shared_ptr<Connection<T>> ptr : this->inCon)
		sum += ptr->get();

	//Activation function
	sum = tanh(sum);

	output = sum;
}

template <class T>
T OutputGraphNode<T>::get() {
	return output;
}

template <class T>
void InputGraphNode<T>::run() {
	for (shared_ptr<Connection<T>> ptr : this->outCon)
		*ptr = input;
}

template <class T>
T InputGraphNode<T>::operator=(T val) {
	input = val;
	return input;
}

//template <class T>
//InputGraphNode<T>::InputGraphNode(int id) {
//	input = T(0);
//}


template <class T>
vector<shared_ptr<Connection<T>>>& GraphNode<T>::getOutCon() {
	return outCon;
}

template <class T>
vector<shared_ptr<Connection<T>>>& GraphNode<T>::getInCon() {
	return inCon;
}

template <class T>
void GraphNode<T>::removeDisconnectedConnections() {
	vector<int> indexs;

	//Check for disconnected inCon
	for (int i = 0; i < inCon.size(); ++i)
		if (inCon[i]->disconnected())indexs.push_back(i);
	//Remove disconnected inCon
	for (int i = indexs.size(); i > 0; --i) {
		inCon.erase(inCon.begin() + indexs.back());
		indexs.pop_back();
	}

	//Check for disconnected outCon
	for (int i = 0; i < outCon.size(); ++i)
		if (outCon[i]->disconnected())indexs.push_back(i);
	//Remove disconnected outCon
	for (int i = indexs.size(); i > 0; --i) {
		outCon.erase(outCon.begin() + indexs.back());
		indexs.pop_back();
	}
}

template <class T>
void GraphNode<T>::run() {
	T sum = T(0);
	for (shared_ptr<Connection<T>> ptr : this->inCon)
		sum += ptr->get();

	//Activation functions
	sum = tanh(sum);
	//if (sum < T(0))sum = T(0);
	//sum = log(sum + 1.0);

	for (shared_ptr<Connection<T>> ptr : this->outCon)
		*ptr = sum;
}

template <class T>
void GraphNode<T>::flipBuffer() {
	//Flip all out-going buffer
	for (shared_ptr<Connection<T>> ptr : this->outCon)
		ptr->flipBuffer();
}

template <class T>
void GraphNode<T>::removeOutCon(shared_ptr<Connection<T>> out) {
	int index = 0;
	for (; index < this->outCon.size(); ++index)
		if (this->outCon[index] == out)break;
	if (index < this->outCon.size())
		this->outCon.erase(this->outCon.begin() + index);
}

template <class T>
void GraphNode<T>::addOutCon(shared_ptr<Connection<T>> out) {
	this->outCon.push_back(out);
}

template <class T>
void GraphNode<T>::removeInCon(shared_ptr<Connection<T>> in) {
	int index = 0;
	for (; index < this->inCon.size(); ++index)
		if (this->inCon[index] == in)break;
	if (index < this->inCon.size())
		this->inCon.erase(this->inCon.begin() + index);
}

template <class T>
void GraphNode<T>::addInCon(shared_ptr<Connection<T>> in) {
	this->inCon.push_back(in);
}

template <class T>
int GraphNode<T>::getId() {
	return id;
}

template <class T>
void GraphNode<T>::setId(int id) {
	this->id = id;
}

template <class T>
GraphNode<T>::GraphNode(int id) {
	this->id = id;
}

template <class T>
void Connection<T>::writeToFile(fstream& out) {
	//Input node id
	out.write(reinterpret_cast<char*>(&inNodeId), sizeof(int));

	//Output node id
	out.write(reinterpret_cast<char*>(&outNodeId), sizeof(int));

	//Weight
	out.write(reinterpret_cast<char*>(&weight), sizeof(T));

	//ABuffer
	out.write(reinterpret_cast<char*>(&ABuffer), sizeof(T));

	//BBuffer
	out.write(reinterpret_cast<char*>(&BBuffer), sizeof(T));

	//BufferState
	out.write(reinterpret_cast<char*>(&useABuffer), sizeof(bool));
}

template <class T>
bool Connection<T>::disconnected() {
	return (inNodeId == -1 || outNodeId == -1);
}

template <class T>
void Connection<T>::removeOutNodeId() {
	this->outNodeId = -1;
}

template <class T>
void Connection<T>::setOutNodeId(int outNodeId) {
	this->outNodeId = outNodeId;
}

template <class T>
int Connection<T>::getOutNodeId() {
	return outNodeId;
}

template <class T>
void Connection<T>::removeInNodeId() {
	this->inNodeId = -1;
}

template <class T>
void Connection<T>::setInNodeId(int inNodeId) {
	this->inNodeId = inNodeId;
}

template <class T>
int Connection<T>::getInNodeId() {
	return inNodeId;
}

template <class T>
void Connection<T>::setWeight(T val) {
	this->weight = val;
}

template <class T>
T Connection<T>::getWeight() {
	return this->weight;
}

template <class T>
T Connection<T>::getBBuffer() {
	return BBuffer;
}

template <class T>
T Connection<T>::getABuffer() {
	return ABuffer;
}

template <class T>
T Connection<T>::get() {
	if (useABuffer)
		return weight * BBuffer;
	else
		return weight * ABuffer;
}

template <class T>
void Connection<T>::setBufferState(bool bufferState) {
	useABuffer = bufferState;
}

template <class T>
bool Connection<T>::getBufferState() {
	return useABuffer;
}

template <class T>
void Connection<T>::operator=(T val) {
	if (useABuffer)
		ABuffer = val;
	else
		BBuffer = val;
}

template <class T>
void Connection<T>::flipBuffer() {
	useABuffer = !useABuffer;
}

template <class T>
Connection<T>::Connection(int inNodeId, int outNodeId, T weight, T ABuffer, T BBuffer, bool useABuffer) {
	this->useABuffer = useABuffer;
	this->weight = weight;
	this->ABuffer = ABuffer;
	this->BBuffer = BBuffer;
	this->inNodeId = inNodeId;
	this->outNodeId = outNodeId;
}

template <class T>
Connection<T>::Connection() {
	useABuffer = false;
	weight = T(1);
	ABuffer = T(0);
	BBuffer = T(0);
	inNodeId = -1;
	outNodeId = -1;
}


#endif
