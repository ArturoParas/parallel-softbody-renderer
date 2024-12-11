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


#include <chrono>
#include <thread>

#include <circle.hpp>
#include <vec2.hpp>
#include <solver.hpp>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

#define MOVE_SPEED 5.f
#define MOUSE_SENSITIVITY 4.f

#define NUM_CIRCLES 2
#define CIRCLE_ENTRY_SIZE 6

#define NUM_SPRINGS 1
#define SPRING_ENTRY_SIZE 3

#define DT 0.1

void solver_update(const softbody_sim::SolverInfo & solver_info, void* circles, void* springs);
// void initialize_springs(const softbody_sim::SolverInfo & solver_info, uint16_t *springs_host, uint16_t *springs_device);
// void host_test_spring_buffer(uint16_t *springs_device);

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
    /** TODO: Why is Shaders dir in src? */
    Shader shader(ReadTextFile("../src/Shaders/vertex.glsl"),ReadTextFile("../src/Shaders/fragment.glsl"));
    shader.Use();
    shader.SetFloatParam("ambientStrength",0.5f);
    shader.SetVec3Param("lightColor",glm::vec3(1.f));

    //Models
    Model model("../tests/particle.fbx");

    //Objects

    std::vector<Object> objects;

    for(int i = 0; i < NUM_CIRCLES; i++){
        objects.emplace_back(&model,glm::vec3(5.f*i,0.f,0.f),glm::vec3(0.f),glm::vec3(2.5f));
    }

    //Solver
    softbody_sim::SolverInfo solver(100,100,NUM_CIRCLES,NUM_SPRINGS,2.5f,2.f,1,DT);
    
    //Circles
    float* circles = (float*)malloc(sizeof(float) * NUM_CIRCLES * CIRCLE_ENTRY_SIZE);

    for(int i = 0; i < NUM_CIRCLES; i++){

        circles[CIRCLE_ENTRY_SIZE*i+0] = (i+1) * 16.f;
        circles[CIRCLE_ENTRY_SIZE*i+1] = 10.f;
        circles[CIRCLE_ENTRY_SIZE*i+2] = (i+1) * 16.f;
        circles[CIRCLE_ENTRY_SIZE*i+3] = 10.f;
        circles[CIRCLE_ENTRY_SIZE*i+4] = (i+1) * 16.f;
        circles[CIRCLE_ENTRY_SIZE*i+5] = 10.f;
    }


    

    //Springs

    uint16_t* springs = (uint16_t*)malloc(sizeof(uint16_t) * NUM_SPRINGS * SPRING_ENTRY_SIZE);

    //for now
    springs[CIRCLE_ENTRY_SIZE*0+0] = 0; //endpoint A
    springs[CIRCLE_ENTRY_SIZE*0+1] = 1; //endpoint B
    springs[CIRCLE_ENTRY_SIZE*0+2] = 12; //Resting Length

    //Camera
    Camera camera(glm::vec3(10.f,0.f,70.f));

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

        

        solver_update(solver, (void*)circles, (void*)springs);


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.Use();
        shader.SetMat4Param("projection",camera.GetProjectionMatrix((float)window.getSize().x,(float)window.getSize().y));
        shader.SetMat4Param("view",camera.GetViewMatrix());
        shader.SetVec3Param("lightPos",camera.position);

        
        for(int i = 0 ; i < NUM_CIRCLES; i++){

            Object obj = objects[i];
            obj.position.x = circles[CIRCLE_ENTRY_SIZE*i + 2];
            obj.position.z = circles[CIRCLE_ENTRY_SIZE*i + 3];

            
            // std::cout << i << ": "<<   obj.position.x<<" "<<  obj.position.z<< std::endl;

            obj.Draw(shader,glm::vec3(0.4f,0.4f,0.8f));
        }

        window.display();
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(50)));

    }

    free(circles);
    // freeSprings(springs);

    return 0;
}


