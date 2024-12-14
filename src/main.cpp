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

#define MAX_RDONLY_NEIGHBORS 3
#define MAX_NEIGHBORS_PER_CIRCLE 3

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

#define THREADS_PER_BLOCK 4
#define NUM_BLOCKS 2

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

// void solver_update(float* host_curr_circles, float* device_curr_circles, float* device_prev_circles, uint16_t* device_neighbor_indices, uint16_t* device_neighbor_map, softbody_sim::SolverInfo & solver_info);
// void solver_update_device(float* host_curr_circles, float* device_curr_circles, float* device_prev_circles, uint16_t* device_neighbor_indices, uint16_t* device_neighbor_map, softbody_sim::SolverInfo & solver_info);
void solver_setup(const softbody_sim::SolverInfo& solver_info);
void solver_trivial(float* h_curr_circles);

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

    int rest_len;
    int threads_per_block;
    int num_blocks;
    int max_pts;
    int max_nbors_per_pt;
    int max_nbors_per_block;
    int max_nbors;
    int max_rdonly_per_block;
    int max_rdonly;

    input_file >> rest_len;
    input_file >> threads_per_block;
    input_file >> num_blocks;
    input_file >> max_pts;
    input_file >> max_nbors_per_pt;
    input_file >> max_nbors_per_block;
    input_file >> max_nbors;
    input_file >> max_rdonly_per_block;
    input_file >> max_rdonly;

    float* host_curr_pts = (float*)malloc(3 * max_pts * sizeof(float));
    uint8_t* pt_indicators = (uint8_t*)malloc(max_pts * sizeof(uint8_t));
    uint16_t* host_rdonly_nbors = (uint16_t*)malloc(max_rdonly * sizeof(uint16_t));
    uint16_t* host_nbor_map = (uint16_t*)malloc(max_nbors * sizeof(uint16_t));

    for (int pt_idx = 0; pt_idx < max_pts; pt_idx += 3) {
      input_file >> host_curr_pts[pt_idx + 0];
      input_file >> host_curr_pts[pt_idx + 1];
      input_file >> host_curr_pts[pt_idx + 2];
    }
    for (int pt_idx = 0; pt_idx < max_pts; pt_idx++) {
      input_file >> pt_indicators[pt_idx];
    }
    for (int rdonly_idx; rdonly_idx < max_rdonly; rdonly_idx++) {
      input_file >> host_rdonly_nbors[rdonly_idx];
    }
    for (int nbor_idx; nbor_idx < max_nbors; nbor_idx++) {
      input_file >> host_nbor_map[nbor_idx];
    }

    float3* device_curr_pts;
    float3* device_prev_pts;
    uint16_t* device_rdonly_nbors;
    uint16_t* device_nbor_map;

    cudaCheckError(cudaMalloc(&device_curr_pts, max_pts * sizeof(float3)));
    cudaCheckError(cudaMalloc(&device_prev_pts, max_pts * sizeof(float3)));
    cudaCheckError(cudaMalloc(&device_rdonly_nbors, max_rdonly * sizeof(uint16_t)));
    cudaCheckError(cudaMalloc(&device_nbor_map, max_nbors * sizeof(uint16_t)));
    
    cudaCheckError(cudaMemcpy(device_curr_pts, host_curr_pts, max_pts * sizeof(float3),cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_prev_pts, host_curr_pts, max_pts * sizeof(float3),cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_rdonly_nbors, host_rdonly_nbors, max_rdonly * sizeof(uint16_t),cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_nbor_map, host_nbor_map, max_nbors * sizeof(uint16_t),cudaMemcpyHostToDevice));

    free(host_rdonly_nbors);
    free(host_nbor_map);

    //Solver
    softbody_sim::SolverInfo solver_info;
    solver_info.spring_rest_length = rest_len;
    solver_info.num_blocks = num_blocks;
    solver_info.threads_per_block = threads_per_block;
    solver_info.max_pts = max_pts;
    solver_info.max_nbors_per_pt = max_nbors_per_pt;
    solver_info.max_nbors_per_block = max_nbors_per_block;
    solver_info.max_nbors = max_nbors;
    solver_info.max_rdonly_per_block = max_rdonly_per_block;
    solver_info.max_rdonly = max_rdonly;
    solver_setup(solver_info);

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

    for(uint32_t i = 0; i < num_blocks * THREADS_PER_BLOCK; i++){

        if(pt_indicators[i] == 0){
            
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

        // solver_update_device(host_curr_circles,device_curr_circles,device_prev_circles,
        //                      device_neighbor_indices,device_neighbor_map,solver_info);
        solver_trivial(host_curr_pts);

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
            
            // if (i != 0) {
              // std::cout << i << ": " << obj.position.x << " " << obj.position.z << std::endl;
            // }

            obj.Draw(shader,glm::vec3(0.4f,0.4f,0.8f));
        }

        window.display();
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(15)));

    }

    free(host_curr_circles);
    free(pt_indicators);

    return 0;
}


