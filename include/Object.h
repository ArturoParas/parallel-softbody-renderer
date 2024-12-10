#pragma once

#include <Shader.h>
#include <Model.h>

#include <glm/glm.hpp>

class Object
{
    public:

        Object(const Model* model,glm::vec3 position = {},glm::vec3 rotation = {},glm::vec3 scale = glm::vec3(1.f));

        glm::mat4 GetTransformationMatrix();

        void SetPosition(float x, float y, float z);

        void Draw(Shader & shader,glm::vec3 color);

        const Model* model;
        glm::vec3 position,rotation,scale;


    private:
};

