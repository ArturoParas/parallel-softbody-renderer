
output: main.o mesh.o shader.o camera.o object.o model.o vec2.o circle.o solver.o cuda_solver.o
	g++ main.o mesh.o shader.o camera.o object.o model.o vec2.o circle.o solver.o cuda_solver.o -o sfml-app -g -L$(shell pwd)/../lib -L/usr/local/cuda-11.7/lib64/ -lsfml-graphics -lsfml-window -lsfml-system -lGL -lGLEW -lglm -lassimp -lcudart

main.o: main.cpp
	g++ -c main.cpp -I$(shell pwd)/../include

mesh.o: mesh.cpp
	g++ -c mesh.cpp -I$(shell pwd)/../include

shader.o: shader.cpp
	g++ -c shader.cpp -I$(shell pwd)/../include

camera.o: camera.cpp
	g++ -c camera.cpp -I$(shell pwd)/../include

object.o: object.cpp
	g++ -c object.cpp -I$(shell pwd)/../include

model.o: model.cpp
	g++ -c model.cpp -I$(shell pwd)/../include

solver.o: solver.cpp
	g++ -c solver.cpp -I$(shell pwd)/../include

circle.o: circle.cpp
	g++ -c circle.cpp -I$(shell pwd)/../include

vec2.o: vec2.cpp
	g++ -c vec2.cpp -I$(shell pwd)/../include

cuda_solver.o : cuda_solver.cu
	nvcc -c cuda_solver.cu -I$(shell pwd)/../include -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

clean:
	rm *.o sfml-app