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
#include <cuda_runtime.h>

#define MOVE_SPEED 50.f
#define MOUSE_SENSITIVITY 4.f

#define MAX_RDONLY_NEIGHBORS 380
#define MAX_NEIGHBORS_PER_CIRCLE 26

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

#define THREADS_PER_BLOCK 320

void solver_update(float* host_curr_circles, float* device_curr_circles, float* device_prev_circles, uint16_t* device_neighbor_indices, uint16_t* device_neighbor_map, softbody_sim::SolverInfo & solver_info);


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

    //Read Input

    std::string input_file_name = "../inputs/sphere.txt";
    std::fstream input_file(input_file_name, std::ios_base::in);

    uint32_t num_blocks;

    input_file >> num_blocks;

    float* host_curr_circles = (float*)malloc(3*num_blocks*THREADS_PER_BLOCK*sizeof(float));
    uint8_t* active_circles = (uint8_t*)malloc(num_blocks*THREADS_PER_BLOCK*sizeof(uint8_t));
    uint16_t* host_neighbor_indices = (uint16_t*)malloc(num_blocks*MAX_RDONLY_NEIGHBORS*sizeof(uint16_t));
    uint16_t* host_neighbor_map = (uint16_t*)malloc(num_blocks*THREADS_PER_BLOCK*MAX_NEIGHBORS_PER_CIRCLE*sizeof(uint16_t));

    uint32_t circles_in_block;
    float circ_x,circ_y,circ_z;
    uint32_t idx_cc= 0;
    uint32_t idx_ac= 0;
    for(uint32_t b=0; b < num_blocks; b++){

        input_file >> circles_in_block;

        for(uint32_t i=0; i < circles_in_block;i++){

            input_file >> circ_x;
            input_file >> circ_y;
            input_file >> circ_z;

            host_curr_circles[idx_cc+0] = circ_x;
            host_curr_circles[idx_cc+1] = circ_y;
            host_curr_circles[idx_cc+2] = circ_z;
            idx_cc+=3;

            active_circles[idx_ac] = 1;
            idx_ac++;
        }
        for(uint32_t i=0; i < THREADS_PER_BLOCK - circles_in_block; i++){

            host_curr_circles[idx_cc+0] = 0.f;
            host_curr_circles[idx_cc+1] = 0.f;
            host_curr_circles[idx_cc+2] = 0.f;
            idx_cc+=3;

            active_circles[idx_ac] = 0;
            idx_ac++;
        }

    }

    uint32_t idx_hni=0;
    uint32_t rdonly_circs;
    uint16_t neighbor;
    for(uint32_t b = 0; b < num_blocks; b++){

        input_file >> rdonly_circs; 

        for(uint32_t i=0; i < rdonly_circs; i++){

            input_file >> neighbor;

            host_neighbor_indices[idx_hni] = neighbor;
            idx_hni++;
        }

        for(uint32_t i=0; i < MAX_RDONLY_NEIGHBORS - rdonly_circs; i++){

            host_neighbor_indices[idx_hni] = 0;
            idx_hni++;
        }
    }

    uint32_t idx_hnm=0;
    uint32_t mapped_to;
    for(uint32_t b=0; b < num_blocks; b++){

        for(uint32_t i=0; i < MAX_NEIGHBORS_PER_CIRCLE*THREADS_PER_BLOCK; i++){

            input_file >> mapped_to;

            host_neighbor_map[idx_hnm] = mapped_to;
            idx_hnm++;
        }
    }


    float* device_curr_circles;
    float* device_prev_circles;
    uint16_t* device_neighbor_indices;
    uint16_t* device_neighbor_map;

    cudaMalloc(&device_curr_circles, 3*num_blocks*THREADS_PER_BLOCK*sizeof(float));
    cudaMalloc(&device_prev_circles, 3*num_blocks*THREADS_PER_BLOCK*sizeof(float));
    cudaMalloc(&device_neighbor_indices, num_blocks*MAX_RDONLY_NEIGHBORS*sizeof(uint16_t));
    cudaMalloc(&device_neighbor_map, num_blocks*THREADS_PER_BLOCK*MAX_NEIGHBORS_PER_CIRCLE*sizeof(uint16_t));

    cudaMemcpy(device_curr_circles, host_curr_circles,  3*num_blocks*THREADS_PER_BLOCK*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_prev_circles, host_curr_circles,  3*num_blocks*THREADS_PER_BLOCK*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbor_indices, host_neighbor_indices,  num_blocks*MAX_RDONLY_NEIGHBORS*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_neighbor_map, host_neighbor_map,  num_blocks*THREADS_PER_BLOCK*MAX_NEIGHBORS_PER_CIRCLE*sizeof(uint16_t), cudaMemcpyHostToDevice);
    
    free(host_neighbor_indices);
    free(host_neighbor_map);


    //Solver

    softbody_sim::SolverInfo solver_info;


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

    for(int i = 0; i < num_blocks * THREADS_PER_BLOCK; i++){

        if(active_circles[i] == 0){
            continue;
        }

        objects.emplace_back(&model,glm::vec3(5.f*i,0.f,0.f),glm::vec3(0.f),glm::vec3(solver_info.circle_radius),i);
    }

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

        solver_update(host_curr_circles,device_curr_circles,device_prev_circles,
                      device_neighbor_indices,device_neighbor_map,solver_info);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.Use();
        shader.SetMat4Param("projection",camera.GetProjectionMatrix((float)window.getSize().x,(float)window.getSize().y));
        shader.SetMat4Param("view",camera.GetViewMatrix());
        shader.SetVec3Param("lightPos",camera.position);

        
        for(uint32_t i = 0 ; i < objects.size(); i++){

            Object& obj = objects[i];
            obj.position.x = host_curr_circles[3*obj.tag+0];
            obj.position.y = host_curr_circles[3*obj.tag+1];
            obj.position.z = host_curr_circles[3*obj.tag+2];
            
            // std::cout << i << ": "<<   obj.position.x<<" "<<  obj.position.z<< std::endl;

            obj.Draw(shader,glm::vec3(0.4f,0.4f,0.8f));
        }

        window.display();
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(15)));

    }

    free(host_curr_circles);
    free(active_circles);

    return 0;
}


