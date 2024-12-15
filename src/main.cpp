//Built-In Libraries
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <getopt.h>

//External Libraries
#include <SFML/Window.hpp>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

//Project Header Files
#include <Mesh.h>
#include <Shader.h>
#include <Camera.h>
#include <Object.h>
#include <Model.h>
#include "../include/cuda_solver.h"

#define MOVE_SPEED 50.f
#define MOUSE_SENSITIVITY 4.f

#define MAX_RDONLY_NEIGHBORS 3
#define MAX_NEIGHBORS_PER_CIRCLE 3

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

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

int main(int argc, char* argv[])
{
    std::string input_file_name = "../inputs/cube.txt";
    GlobalConstants h_params;
    /** TODO: Good debugging params, delete later */
    h_params.set_dt_and_intermediate_steps(0.01, 1);
    h_params.g = -9.81;
    h_params.width = 110;
    h_params.height = 110;
    h_params.depth = 110;
    h_params.spring_k = 100;
    /** TODO: Come back to damping, something seems off... */
    h_params.spring_damp = 1;
    h_params.particle_rad = 1;
    int benchmark_iters = 100;

    int opt;
    char* end;
    while ((opt = getopt(argc, argv, "s:k:c:g:w:h:d:t:i:r:m:")) != EOF) {
        switch(opt) {
        case 's':
            if (*optarg == 'c') {
                input_file_name = "../inputs/cube.txt";
            } else if (*optarg == 'p') {
                input_file_name = "../inputs/pyramid.txt";
            } else if (*optarg == 's') {
                input_file_name = "../inputs/sphere.txt";
            }
            break;
        case 'k':
            h_params.spring_k = strtod(optarg, &end);
            break;
        case 'c':
            h_params.spring_damp = strtod(optarg, &end);
            break;
        case 'g':
            h_params.g = strtod(optarg, &end);
            break;
        case 'w':
            h_params.width = strtod(optarg, &end);
            break;
        case 'h':
            h_params.height = strtod(optarg, &end);
            break;
        case 'd':
            h_params.depth = strtod(optarg, &end);
            break;
        case 't':
            h_params.set_dt(strtod(optarg, &end));
            break;
        case 'i':
            h_params.set_intermediate_steps(strtol(optarg, &end, 10));
            break;
        case 'r':
            h_params.set_particle_rad(strtod(optarg, &end));
            break;
        case 'm':
            benchmark_iters = strtol(optarg, &end, 10);
            break;
        default:
            break;
        }
    }

    // Read Input
    std::fstream input_file(input_file_name, std::ios_base::in);
    /** TODO: Damping too high, make it with respect to time instead of frames, but how? */

    input_file >> h_params.spring_rest_len;
    input_file >> h_params.particles_per_block;
    input_file >> h_params.num_blocks;
    input_file >> h_params.max_particles;
    input_file >> h_params.max_nbors_per_particle;
    input_file >> h_params.max_nbors_per_block;
    input_file >> h_params.max_nbors;
    input_file >> h_params.max_rdonly_per_block;
    input_file >> h_params.max_rdonly;
    int rad;
    input_file >> rad;
    h_params.height = 2 * rad + h_params.particle_diameter + 10;
    h_params.width = 2 * rad + h_params.particle_diameter + 10;
    h_params.depth = 2 * rad + h_params.particle_diameter + 10;

    float* h_curr_particles = (float*)malloc(3 * h_params.max_particles * sizeof(float));
    bool* particle_indicators = (bool*)malloc(h_params.max_particles * sizeof(bool));
    int16_t* h_rdonly_nbors = (int16_t*)malloc(h_params.max_rdonly * sizeof(int16_t));
    int16_t* h_nbor_map = (int16_t*)malloc(h_params.max_nbors * sizeof(int16_t));

    for (int particle_idx = 0; particle_idx < 3 * h_params.max_particles; particle_idx += 3) {
        input_file >> h_curr_particles[particle_idx + 0];
        input_file >> h_curr_particles[particle_idx + 1];
        input_file >> h_curr_particles[particle_idx + 2];
    }
    for (int particle_idx = 0; particle_idx < h_params.max_particles; particle_idx++) {
        input_file >> particle_indicators[particle_idx];
    }
    for (int rdonly_idx = 0; rdonly_idx < h_params.max_rdonly; rdonly_idx++) {
        input_file >> h_rdonly_nbors[rdonly_idx];
    }
    for (int nbor_idx = 0; nbor_idx < h_params.max_nbors; nbor_idx++) {
        input_file >> h_nbor_map[nbor_idx];
    }

    solver_setup(h_params, h_curr_particles, h_rdonly_nbors, h_nbor_map);

    free(h_rdonly_nbors);
    free(h_nbor_map);

    //Environment
    sf::ContextSettings contextsettings;
    contextsettings.attributeFlags = sf::ContextSettings::Default;
    contextsettings.majorVersion = 4;
    contextsettings.minorVersion = 6;
    contextsettings.depthBits = 24;
    /** TODO: Do the width and height need to be known at compile time? */
    sf::Window window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "3D OpenGL",
        sf::Style::Default,contextsettings);
    sf::Clock clock;

    int glewInitResult;
    if((glewInitResult = glewInit()) != GLEW_OK){
        std::cerr << "Failed to Initialize Glew (" << glewInitResult  << ")\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    //Shaders
    Shader shader(ReadTextFile("../src/Shaders/vertex.glsl"),
        ReadTextFile("../src/Shaders/fragment.glsl"));
    shader.Use();
    shader.SetFloatParam("ambientStrength", 0.5f);
    shader.SetVec3Param("lightColor", glm::vec3(1.f));

    //Models
    Model model("../tests/particle.fbx");

    //Objects
    std::vector<Object> objects;

    for (int i = 0; i < h_params.max_particles; i++) {
        if (particle_indicators[i]) {
            objects.emplace_back(&model, glm::vec3(5.f * i, 0.f, 0.f), glm::vec3(0.f),
                glm::vec3(h_params.particle_rad), i);
        }
    }

    //Camera
    Camera camera(glm::vec3(10.f, 0.f, 70.f));

    //Main Loop
    bool isFirstMouse = true;
    sf::Vector2i lastMousePos;
    double compute_time = 0;
    double draw_time = 0;
    double total_time = 0;
    double kernel_time = 0;
    int benchmark_idx = 0;
    std::cout << "Running for " << benchmark_iters << " iterations." << std::endl;
    const auto total_start = std::chrono::steady_clock::now();
    while (benchmark_idx < benchmark_iters) {
        benchmark_idx++;
        float dt = clock.restart().asSeconds();

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::Resized) {
                glViewport(0, 0, window.getSize().x, window.getSize().y);
            }
        }

        camera.UpdateDirectionVectors();
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
            camera.position += camera.Forward() * MOVE_SPEED * dt;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
            camera.position -= camera.Right() * MOVE_SPEED * dt;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
            camera.position -= camera.Forward() * MOVE_SPEED * dt;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
            camera.position += camera.Right() * MOVE_SPEED * dt;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
          camera.position += camera.Up() * MOVE_SPEED * dt;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
          camera.position -= camera.Up() * MOVE_SPEED * dt;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) {
            window.close();
        }

        if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
            if (isFirstMouse) {
                lastMousePos = sf::Mouse::getPosition(window);
                isFirstMouse = false;
                window.setMouseCursorVisible(false);
            } else {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                int xOffset = mousePos.x - lastMousePos.x;
                int yOffset = lastMousePos.y - mousePos.y;

                camera.yaw += xOffset * MOUSE_SENSITIVITY * dt;
                camera.pitch += yOffset * MOUSE_SENSITIVITY * dt;

                sf::Mouse::setPosition(lastMousePos, window);
            }
        } else {
            isFirstMouse = true;
            window.setMouseCursorVisible(true);
        }

        const auto compute_start = std::chrono::steady_clock::now();
        kernel_time += solver_update(h_params, h_curr_particles);
        const auto compute_end = std::chrono::steady_clock::now();
        compute_time += std::chrono::duration_cast<std::chrono::duration<double>>(compute_end - compute_start).count();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.Use();
        shader.SetMat4Param("projection",camera.GetProjectionMatrix((float)window.getSize().x,(float)window.getSize().y));
        shader.SetMat4Param("view",camera.GetViewMatrix());
        shader.SetVec3Param("lightPos",camera.position);

        const auto draw_start = std::chrono::steady_clock::now();
        for (int i = 0 ; i < objects.size(); i++){
            Object& obj = objects[i];
            obj.position.x = h_curr_particles[3 * obj.tag + 0];
            obj.position.y = h_curr_particles[3 * obj.tag + 1];
            obj.position.z = h_curr_particles[3 * obj.tag + 2];

            obj.Draw(shader, glm::vec3(0.4f, 0.4f, 0.8f));
        }

        window.display();
        const auto draw_end = std::chrono::steady_clock::now();
        draw_time += std::chrono::duration_cast<std::chrono::duration<double>>(draw_end - draw_start).count();
        // std::this_thread::sleep_for(std::chrono::milliseconds((int)(h_params.dt * 1000)));
    }
    if (window.isOpen()) {
        window.close();
    }
    const auto total_end = std::chrono::steady_clock::now();
    total_time = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start).count();
    std::cout << std::endl;
    std::cout << "Kernel time = " << kernel_time << " s" << std::endl;
    std::cout << "Compute time = " << compute_time << " s" << std::endl;
    std::cout << "Draw time = " << draw_time << " s" << std::endl;
    std::cout << "Total time = " << total_time << " s" << std::endl;
    std::cout << std::endl;
    std::cout << "Kernel time per iteration = " << kernel_time / benchmark_iters << " s/iter" << std::endl;
    std::cout << "Compute time per iteration = " << compute_time / benchmark_iters << " s/iter" << std::endl;
    std::cout << "Draw time per iteration = " << draw_time / benchmark_iters << " s/iter" << std::endl;
    std::cout << "Total time per iteration = " << total_time / benchmark_iters << " s/iter" << std::endl;

    free(h_curr_particles);
    free(particle_indicators);
    solver_free(h_params);

    return 0;
}
