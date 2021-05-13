make: clean test.cpp T_EvolutionGraphNN.h
	g++ test.cpp -o test -lpthread

clean:
	rm -f test
	rm -f AND_GATE.TEvoGNN
	rm -f andgate.dot andgate.svg
	rm -f orgate.dot orgate.svg
	rm -f notgate.dot notgate.svg
	
run: test
	./test
	dot -Tsvg notgate.dot -o notgate.svg
	dot -Tsvg andgate.dot -o andgate.svg
	dot -Tsvg orgate.dot -o orgate.svg
