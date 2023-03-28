make: clean test.cpp T_EvolutionGraphNN.h
	g++ test.cpp -o test -lpthread -std=c++20

clean:
	rm -f test
	rm -f AND_GATE.TEvoGNN
	rm -f andgate.dot andgate.svg
	rm -f orgate.dot orgate.svg
	rm -f notgate.dot notgate.svg
	rm -f aNetwork.dot aNetwork.svg
	rm -f bNetwork.dot bNetwork.svg
	rm -f cNetwork.dot cNetwork.svg
	
run: test
	./test
	dot -Tsvg notgate.dot -o notgate.svg
	dot -Tsvg andgate.dot -o andgate.svg
	dot -Tsvg orgate.dot -o orgate.svg
	dot -Tsvg aNetwork.dot -o aNetwork.svg
	dot -Tsvg bNetwork.dot -o bNetwork.svg
	dot -Tsvg cNetwork.dot -o cNetwork.svg
