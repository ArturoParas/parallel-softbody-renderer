#pragma once

#include <glm/glm.hpp>

#include <string>

class Shader
{

public:
    Shader(const std::string &vertexShaderFilename, const std::string &fragmentShaderFilename);
    void Use();
    void SetFloatParam(const std::string &paramName,float val);
    void SetVec3Param(const std::string &paramName,glm::vec3 val);
    void SetMat4Param(const std::string &paramName,glm::mat4 val);

private:
    uint32_t programID;
};