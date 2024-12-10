
output: main.o mesh.o shader.o camera.o object.o model.o vec2.o circle.o solver.o
	g++ main.o mesh.o shader.o camera.o object.o model.o vec2.o circle.o solver.o -o sfml-app -g -L$(shell pwd)/../lib -lsfml-graphics -lsfml-window -lsfml-system -lGL -lGLEW -lglm -lassimp

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


clean:
	rm *.o sfml-app