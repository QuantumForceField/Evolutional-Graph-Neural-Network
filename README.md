# Evolutional-Graph-Neural-Network
T_EvolutionGraphNN.h is a C++ library which simulates directed graph neural networks and their evolution processes.

***

## Table of Content
- [Introductions](#Introductions)
    - [Base Classes](#Base-Classes)
        - [GraphNode](#GraphNode)
        - [Connection](#Connection)
    - [EvolutionGNN](#EvolutionGNN)
- [Requirements](#Requirements)
- [Examples](#Examples)
    - [Basic Usages](#Basic-Usages)
    - [Logic Gates](#Logic-Gates)
        - [NOT Gate](#NOT-Gate)
        - [AND Gate](#AND-Gate)
        - [OR Gate](#OR-Gate)
    - [Oscillator](#Oscillator)
    - [Memory Cell](#Memory-Cell)

***

## Introductions
This library is created to simulate graph neural networks in a time-step manner. The library allowed users to create, configure, run, save, load and visualize graph neural networks, as well as simulates evolution processes of the network by inheritance, and mutations.

As this library simulates graph neural networks in a time-based manner, neural networks with cycles are allowed to exist.

### Base Classes

In the library, neurons and connections between neurons are implemented as **GraphNode** and **Connection** respectively.

#### GraphNode
**GraphNode** represents an abstract concept of neurons in the graph neural network. It manages a list of in-coming connections and out-going connections.

The **GraphNode** is like any common neurons in artificial neural networks, summing all inputs and pass it through an activation function to produce an output.

<p align="center">
<img alt="GraphNode representation" src="./img/graphNode.svg">
</p>

Currently, the activation function for every neurons in T_EvolutionGraphNN is *tanh*.
By calling the function `run()`, **GraphNode** performs above calculations, and stores the output in all of it's out-going connections.

#### Connection
**Connection** represents an abstract concept of connections between neurons, or *"Axons"* in real neurons. In T_EvolutionGraphNN, each **Conncetion** only connects between 2 neurons, **Connection** connecting back to the same neuron is allowed.

Similar to popular neural network concepts, a weight is added to each connection. Differs from common neural networks, the calculation result of neuron is actually stored in **Connection** in T_EvolutionGraphNN.

To allow parallel executions and to avoid problems created by cyclic graph, each **Connection** internally maintains 2 buffers to store calculated result from neuron. At any time step, write operation will be performed on one of the buffer, and any read operation will only be performed on another one. It is after `flipBuffer()` been called that both buffers will swap their roles.

Calling `get()` in **Connection** does not directly returns a value stored in the buffer, but returns *weight \* value* instead.

### EvolutionGNN
**EvolutionGNN** is the class that maintains all **GraphNode** and **Connection** in a single graph neural network. For general users, knowledge about how to use this class is very enough.

To initialize the **EvolutionGNN**, one should know the total amount of inputs and outputs, which cannot be changed unless initialized again. We can initialize **EvolutionGNN** using the constructor, like `EvolutionGNN<float> egnn(9, 3);` creates an **EvolutionGNN** instance with 9 inputs and 3 outputs. We can also initialize the **EvolutionGNN** later by using the function `initialize()`, like
```cpp
EvolutionGNN<float> egnn;

//Initialize in a later stage
egnn.initialize(9, 3);
```

***

## Requirements

### Linux
- [x] g++
    - Install **g++** by running ```sudo apt install g++ -y``` in linux terminal.
- [x] Make
    - Install **make** by running ```sudo apt install make -y``` in linux terminal.
- [x] Graphviz
    - Install **graphviz** by running ```sudo apt install graphviz -y``` in linux terminal.

***

## Examples
### Basic Usages
To use the EvolutionGNN library, we just need to include the header file in our code:
```cpp
#include "T_EvolutionGraphNN.h"
```
To compile the program using **g++**, we need to include the **-lpthread** option since EvolutionGNN supports multi-thread execution.
```bash
g++ your_cpp_file.cpp -o your_desired_executable_name -lpthread
```

The test code of EvolutionGraphNN is included in the [src](./src) folder.

To compile and run the test code, follow these steps:

1. Open the terminal in the **Evolutional-Graph-Neural-Network** folder.
1. Navigate to the **src** folder in the terminal by typing `cd src/` and hit **Enter**.
1. Compile the executable by typing `make` and hit **Enter** in the terminal.
1. Run the compiled executable and generate .svg of test models by typing `make run` and hit **Enter** in the terminal.


### Logic Gates

In this section, I'll demonstrate that the graph neural network is able to act as logic gates.

Since the activation of the neural network is ***tanh***, we can expect that the output of each neuron will be bounded in **(-1.0, 1.0)**. To take advantage of this feature, we will define **True** as **1**, and **False** as **-1** for the neural network.

***

#### **NOT Gate**
To create a **Not Gate**, we can simply flip the input of a neuron. So the architecture will be like this:

<p align="center"> 
<img src="./img/notgate.svg" alt="Architecture for Not Gate">
</p>

To create such neural network, we should define the network to have **1 input node** and **1 output node**:

```cpp
//Create a Not Gate network
EvolutionGNN<float> notGate;

//Set the network to have 1 input node and 1 output node
notGate.initialize(1, 1);

//Not Gate only requires 1 connection, with a large negative weight
//Since we only have 1 input node, the id of input node will be 0
//Since we only have 1 output node, the id of output node will be (#input_nodes + 0), which is 1
notGate.addConnection(0, 1, -20.0);
```

And that completes the building of **Not Gate** with EvolutionGNN.


To test if the **Not Gate** runs as expected, we make 2 test cases.

The first one with input set to -1.0 (false):
```cpp
//Set input indexed at 0 to -1.0
notGate.setInput(0, -1.0);
```
We then run the notGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    notGate.run();

    //Flip internal buffer
    notGate.flipBuffer();

    //Show the output at index -
    std::cout << notGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 1 1 1 1 1 1 1 1 1
```

Another test case is with input set to 1.0 (true):
```cpp
//Set input indexed at 0 to 1.0
notGate.setInput(0, 1.0);
```
We again run the notGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    notGate.run();

    //Flip internal buffer
    notGate.flipBuffer();

    //Show the output at index 0
    std::cout << notGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 -1 -1 -1 -1 -1 -1 -1 -1 -1
```
These proved that our **Not Gate** works!

***

#### **AND Gate**
To create an **And Gate**, the architecture is a bit more complicated than **Not Gate**. Which looks like this:

<p align="center"> 
<img src="./img/andgate.svg" alt="Architecture for And Gate">
</p>

One thing to note is that **node #3** is not a normal node. It acts as a *bias*, and constantly output 1 so that when its multiplied by -60.0 it constantly output -60.0. The reason for this is that we want **And Gate** only turns on (outputs 1) when both inputs are **True** (1) but not when only one of the inputs is **True**.  

To create such neural network, we should define the network to have **2 input nodes** and **1 output node**:

```cpp
//Create a And Gate network
EvolutionGNN<float> andGate;

//Set the network to have 2 input node and 1 output node
andGate.initialize(2, 1);

//And Gate requires 4 connections
//We first create connections from input nodes to output node
//Since we have two input nodes, their node id will be 0 and 1
//Since we have 1 output node, it's node id will be (#input_nodes + 0), which is 2
andGate.addConnection(0, 2, 40.0);
andGate.addConnection(1, 2, 40.0);

//We then create a loop back connect for node 3 and make sure it initials 
//with both buffers set to 1.0 so that the node will always output 1.0
andGate.addConnection(3, 3, 20.0, 1.0, 1.0);

//Finally, we add the connection from node 3 to output node
andGate.addConnection(3, 2, -60.0);
```

And that completes the building of **And Gate** with EvolutionGNN.

To test if the **And Gate** runs as expected, we make 3 test cases.

The first one with both inputs set to -1.0 (false):
```cpp
//Set input indexed at 0 to -1.0
andGate.setInput(0, -1.0);
//Set input indexed at 1 to -1.0
andGate.setInput(1, -1.0);
```
We then run the andGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    andGate.run();

    //Flip internal buffer
    andGate.flipBuffer();

    //Show the output at index 0
    std::cout << andGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

Another test case is with inputs set to -1.0 (false) and 1.0 (true) respectively:
```cpp
//Set input indexed at 0 to -1.0
andGate.setInput(0, -1.0);
//Set input indexed at 1 to 1.0
andGate.setInput(1, 1.0);
```
We again run the notGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    andGate.run();

    //Flip internal buffer
    andGate.flipBuffer();

    //Show the output at index 0
    std::cout << andGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

In the last test case, we set both inputs to 1 (true):
```cpp
//Set input indexed at 0 to 1.0
andGate.setInput(0, 1.0);
//Set input indexed at 1 to 1.0
andGate.setInput(1, 1.0);
```
We again run the notGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    andGate.run();

    //Flip internal buffer
    andGate.flipBuffer();

    //Show the output at index 0
    std::cout << andGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 1 1 1 1 1 1 1 1 1
```
These proved that our **And Gate** works!

***

#### **OR Gate**

To create an **Or Gate**, the architecture is a bit more complicate than **And Gate**:

<p align="center"> 
<img src="./img/orgate.svg" alt="Architecture for Or Gate">
</p>

Again, **node #3** is not a normal node. It acts as a *bias*, and constantly output 1. 

To create such neural network, we should define the network to have **2 input nodes** and **1 output node**:

```cpp
//Create a Not Gate network
EvolutionGNN<float> orGate;

//Set the network to have 2 input node and 1 output node
orGate.initialize(2, 1);

//And Gate requires 8 connections
//We first create connections from input nodes to 2 hidden nodes
//Since we have two input nodes, their node id will be 0 and 1 respectively
//Since we have 3 hidden nodes, their node ids will be >= (#input_nodes + #output_nodes), which is >= 3
//so their node ids are 3, 4 and 5 respectively
//Here we set node 4 and node 5 to accept output from input nodes
orGate.addConnection(0, 4, 20.0);
orGate.addConnection(1, 5, 20.0);

//We then create a loop back connect for node 3 and make sure it initials 
//with both buffers set to 1.0 so that the node will always output 1.0
orGate.addConnection(3, 3, 20.0, 1.0, 1.0);

//Next, we create two conenctions from node 3 to node 4 and node 5 respectively
orGate.addConnection(3, 4, 20.0);
orGate.addConnection(3, 5, 20.0);

//Finally, we add the connection from node 3, node 4 and node 5 to the output node
orGate.addConnection(3, 2, -20.0);
orGate.addConnection(4, 2, 40.0);
orGate.addConnection(5, 2, 40.0);
```

And that completes the building of **Or Gate** with EvolutionGNN.

To test if the **Or Gate** runs as expected, we make 3 test cases.

The first one with both inputs set to -1.0 (false):
```cpp
//Set input indexed at 0 to -1.0
orGate.setInput(0, -1.0);
//Set input indexed at 1 to -1.0
orGate.setInput(1, -1.0);
```
We then run the andGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    orGate.run();

    //Flip internal buffer
    orGate.flipBuffer();

    //Show the output at index 0
    std::cout << orGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

Another test case is with inputs set to -1.0 (false) and 1.0 (true) respectively:
```cpp
//Set input indexed at 0 to -1.0
orGate.setInput(0, -1.0);
//Set input indexed at 1 to 1.0
orGate.setInput(1, 1.0);
```
We again run the notGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    orGate.run();

    //Flip internal buffer
    orGate.flipBuffer();

    //Show the output at index -
    std::cout << orGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 -1 1 1 1 1 1 1 1 1
```

In the last test case, we set both inputs to 1 (true):
```cpp
//Set input indexed at 0 to 1.0
orGate.setInput(0, 1.0);
//Set input indexed at 1 to 1.0
orGate.setInput(1, 1.0);
```
We again run the notGate for 10 times:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run the simulation for 1 time frame
    orGate.run();

    //Flip internal buffer
    orGate.flipBuffer();

    //Show the output at index 0
    std::cout << orGate.getOutput(0) << ' ';
}
std::cout << std::endl;
```
The outcome will looks like this
```
0 -1 1 1 1 1 1 1 1 1
```
These proved that our **Or Gate** works!

***

### Oscillator
Since T_EvolutionGraphNN allows cyclic graphs to exist, creating oscillator can be extremely easy to achieve.

Here is a simple **oscillator** network:
<p align="center">
<img alt="Oscillator design" src="./img/oscillator.svg">
</p>

To create such oscillator, we should first create an EvolutionGNN with 1 input and 1 output:
```cpp
//Create an oscillator
EvolutionGNN<float> oscillator;

//Initialize with 1 input and 1 output
oscillator.initialize(1, 1);
```

Add an additional node to the network:
```cpp
//Add a node
oscillator.addNodes();
```

Connect nodes as shown in the figure:
```cpp
//Input node to node #2
oscillator.addConnection(0, 2, 20.0);

//Node #2 to itself
oscillator.addConnection(2, 2, -40.0);

//Node #2 to output
oscillator.addConnection(2, 1, 20.0);
```

To test the above oscillator, we should first kick-start the oscillation by giving some input to the system.
```cpp
//Set input to 1.0 for 1 time step
oscillator.setInput(0, 1.0);
oscillator.run();
oscillator.flipBuffer();

//Set the input back to 0.0 for the remaining time steps
oscillator.setInput(0, 0.0);
oscillator.run();
oscillator.flipBuffer();
```

We then run the oscillator for 10 time steps:
```cpp
for(int i = 0; i < 10; ++i) {
    //Run for 1 time step
    oscillator.run();

    //Flip internal buffer
    oscillator.flipBuffer();

    //Show current output at index 0
    std::cout << oscillator.getOutput(0) << ' ';
}
std::cout << std::endl;
```

And the result looks like this:
```
1 -1 1 -1 1 -1 1 -1 1 -1
```


In fact, we can create many other types of oscillators as well, here is an example of oscillator that oscillates slower than the previous one:

<p align="center">
<img alt="Slower Oscillator design" src="./img/slowOscillator.svg">
</p>

Codes to create the oscillator above:
```cpp
//Create an oscillator
EvolutionGNN<float> oscillator;

//Initialize with 1 input and 1 output
oscillator.initialize(1, 1);

//Add nodes
oscillator.addNodes(5);

//Add connections
oscillator.addConnection(0, 2, 20.0);
oscillator.addConnection(2, 2, 20.0);
oscillator.addConnection(2, 3, 20.0);
oscillator.addConnection(3, 4, 20.0);
oscillator.addConnection(4, 5, 20.0);
oscillator.addConnection(5, 6, 20.0);
oscillator.addConnection(6, 2, -40.0);
oscillator.addConnection(2, 1, 20.0);

//Set input to 1.0 for 1 time step
oscillator.setInput(0, 1.0);
oscillator.run();
oscillator.flipBuffer();

//Set the input back to 0.0 for the remaining time steps
oscillator.setInput(0, 0.0);
oscillator.run();
oscillator.flipBuffer();

for (int i = 0; i < 20; ++i) {
	//Run for 1 time step
	oscillator.run();

	//Flip internal buffer
	oscillator.flipBuffer();

	//Show current output at index 0
	std::cout << oscillator.getOutput(0) << ' ';
}
std::cout << std::endl;
```
With output:
```
1 1 1 1 1 -1 -1 -1 -1 -1 1 1 1 1 1 -1 -1 -1 -1 -1
```

***

### Memory Cell
Been able to remember things are essential for intelligent systems to develop higher order thinking parts. T_EvolutionGraphNN happens to support it.

Similar to oscillators, memory cells are also created with cyclic graphs, and they can be extremely simple in design too.

Here is an example of an **Erasable Memory Cell**:

<p align="center">
<img alt="Memory Cell design" src="./img/memoryCell.svg">
</p>

The memory cell is constructed with the following code:
```cpp
//Create an memory cell
EvolutionGNN<float> memCell;

//Initialize with 1 input and 1 output
memCell.initialize(1, 1);

//Add 1 node
memCell.addNodes();

//Link input to node# 2
memCell.addConnection(0, 2, 40.0);

//Link node# 2 to itself
memCell.addConnection(2, 2, 20.0);

//Link node# 2 to output
memCell.addConnection(2, 1, 20.0);
```

To write 1.0 into the memory cell:
```cpp
//Write 1.0 into the input
memCell.setInput(0, 1.0);
memCell.run();
memCell.flipBuffer();
//Reset input to 0.0
memCell.setInput(0, 0.0);
memCell.run();
memCell.flipBuffer(); 
```

Watch the output of the memory cell for 10 time steps:
```cpp
for (int i = 0; i < 10; ++i) {
    //Run for 1 time step
    memCell.run();

    //Flip internal buffer
    memCell.flipBuffer();

    //Show current output
    std::cout << memCell.getOutput(0) << ' ';
}
std::cout << std::endl;
```

Which shows:
```
1 1 1 1 1 1 1 1 1 1
```

Next we change the input to -1.0 for 1 time step
```cpp
//Write -1.0 into the input
memCell.setInput(0, -1.0);
memCell.run();
memCell.flipBuffer();
//Reset input to 0.0
memCell.setInput(0, 0.0);
memCell.run();
memCell.flipBuffer(); 
```

Watch the output of the memory cell for the next 10 time steps:
```cpp
for (int i = 0; i < 10; ++i) {
    //Run for 1 time step
    memCell.run();

    //Flip internal buffer
    memCell.flipBuffer();

    //Show current output
    std::cout << memCell.getOutput(0) << ' ';
}
std::cout << std::endl;
```
Which shows:
```
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

These showed that the memory cell is able to remember previous input, and the memory can be erased when new input is given.