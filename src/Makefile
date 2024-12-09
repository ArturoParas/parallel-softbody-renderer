
output: main.o mesh.o shader.o camera.o object.o model.o
	g++ main.o mesh.o shader.o camera.o object.o model.o -o sfml-app -g -L/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/lib -lsfml-graphics -lsfml-window -lsfml-system -lGL -lGLEW -lglm -lassimp

main.o: main.cpp
	g++ -c main.cpp -I/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/include

mesh.o: mesh.cpp
	g++ -c mesh.cpp -I/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/include

shader.o: shader.cpp
	g++ -c shader.cpp -I/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/include

camera.o: camera.cpp
	g++ -c camera.cpp -I/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/include

object.o: object.cpp
	g++ -c object.cpp -I/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/include

model.o: model.cpp
	g++ -c model.cpp -I/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/include


clean:
	rm *.o sfml-app