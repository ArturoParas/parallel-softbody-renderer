//Built-In Libraries
#include <iostream>
#include <fstream>
#include <sstream>

//External Libraries
#include <SFML/Window.hpp>
#include <GL/glew.h>
#include <glm/glm.hpp>

//Project Header Files
#include <Mesh.h>
#include <Shader.h>
#include <Camera.h>
#include <Object.h>
#include <Model.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

#define MOVE_SPEED 5.f
#define MOUSE_SENSITIVITY 1.f

//export LD_LIBRARY_PATH=/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/lib:/usr/lib/
//g++ -c main.cpp -I/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/include
//g++ main.o mesh.o -o sfml-app -L/afs/andrew.cmu.edu/usr10/hflee/private/15418/TensileFlow/minimalrenderer/lib -lsfml-graphics -lsfml-window -lsfml-system -lGL


//I will move this method later 
std::string ReadTextFile(const std::string & filename){

    std::ifstream file(filename);
    if(!file.is_open())
    {
        return "";
    }

    std::stringstream ss;
    ss << file.rdbuf();
    file.close();

    return ss.str();

}

int main()
{

    //Environment
    sf::ContextSettings contextsettings;
    contextsettings.attributeFlags = sf::ContextSettings::Default;
    contextsettings.majorVersion = 4;
    contextsettings.minorVersion = 6;
    contextsettings.depthBits = 24;
    sf::Window window(sf::VideoMode(WINDOW_WIDTH,WINDOW_HEIGHT),"3D OpenGL", sf::Style::Default,contextsettings);
    // sf::Window window(sf::VideoMode(WINDOW_WIDTH,WINDOW_HEIGHT),"3D OpenGL");
    sf::Clock clock;

    int glewInitResult;
    if((glewInitResult = glewInit()) != GLEW_OK){
        std::cerr << "Failed to Initialize Glew (" << glewInitResult  << ")\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    //Shaders
    Shader shader(ReadTextFile("Shaders/vertex.glsl"),ReadTextFile("Shaders/fragment.glsl"));
    shader.Use();
    shader.SetFloatParam("ambientStrength",0.5f);
    shader.SetVec3Param("lightColor",glm::vec3(1.f));

    //Models
    // Model model(Model::MODEL_PRIMITIVE_ICOSPHERE2);
    Model model("../tests/particle.fbx");

    //Objects
    Object object(&model);
    Object object2(&model, glm::vec3(3.f, 0.f,0.f));

    //Camera
    Camera camera(glm::vec3(0.f,0.f,10.f));

    //Main Loop

    bool isFirstMouse = true;
    sf::Vector2i lastMousePos;
    while (window.isOpen())
    {

        float dt = clock.restart().asSeconds();

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed){
                window.close();
            }
            else if(event.type == sf::Event::Resized){
                glViewport(0,0,window.getSize().x,window.getSize().y);
            }

        }

        camera.UpdateDirectionVectors();
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::W))
        {
            camera.position += camera.Forward() * MOVE_SPEED * dt;
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::A))
        {
            camera.position += -camera.Right() * MOVE_SPEED * dt;
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::S))
        {
            camera.position += -camera.Forward() * MOVE_SPEED * dt;
        }
        if(sf::Keyboard::isKeyPressed(sf::Keyboard::D))
        {
            camera.position += camera.Right() * MOVE_SPEED * dt;
        }

        if(sf::Mouse::isButtonPressed(sf::Mouse::Right))
        {
            if(isFirstMouse)
            {   
                lastMousePos = sf::Mouse::getPosition(window);
                isFirstMouse = false;
                window.setMouseCursorVisible(false);
            }
            else
            {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                int xOffset = mousePos.x - lastMousePos.x;
                int yOffset = lastMousePos.y - mousePos.y;

                camera.yaw += xOffset * MOUSE_SENSITIVITY * dt;
                camera.pitch += yOffset * MOUSE_SENSITIVITY * dt;

                sf::Mouse::setPosition(lastMousePos,window);

            }
        }
        else{
            isFirstMouse = true;
            window.setMouseCursorVisible(true);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.Use();
        shader.SetMat4Param("projection",camera.GetProjectionMatrix((float)window.getSize().x,(float)window.getSize().y));
        shader.SetMat4Param("view",camera.GetViewMatrix());
        shader.SetVec3Param("lightPos",camera.position);

        object.Draw(shader,glm::vec3(0.4f,0.4f,0.8f));
        object2.Draw(shader,glm::vec3(0.8f,0.4f,0.4f));

        window.display();

    }

    return 0;
}



