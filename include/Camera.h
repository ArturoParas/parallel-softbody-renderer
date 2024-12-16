#pragma once

#include <glm/glm.hpp>

class Camera
{

    public:

        static const glm::vec3 WorldUp;

        Camera(glm::vec3 position, float pitch=0.f, float yaw=-90.f, float fov=60.f);

        void UpdateDirectionVectors();
        glm::vec3 Forward();
        glm::vec3 Right();
        glm::vec3 Up();


        glm::mat4 GetProjectionMatrix(float width, float height);
        glm::mat4 GetViewMatrix();


        glm::vec3 position;
        float fov, yaw, pitch;


    private:
        glm::vec3 forward,right,up;



};