#include <Shader.h>

#include <glm/gtc/type_ptr.hpp>
#include <GL/glew.h>

#include <iostream>

Shader::Shader(const std::string &vertexCode, const std::string &fragmentCode)
{

    const char* vertexShaderCode = vertexCode.c_str();

    const char* fragmentShaderCode = fragmentCode.c_str();

    char errorlog[1024];
    int success;

    uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader,1,&vertexShaderCode,nullptr);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader,GL_COMPILE_STATUS,&success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader,1024,nullptr,errorlog);
        std::cerr << "Failed to Compile Vertex Shader: " << errorlog;
    }

    uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader,1,&fragmentShaderCode,nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader,GL_COMPILE_STATUS,&success);
    if(!success)
    {
        glGetShaderInfoLog(fragmentShader,1024,nullptr,errorlog);
        std::cerr << "Failed to Compile Fragment shader: " << errorlog;
    }

    programID = glCreateProgram();
    glAttachShader(programID,vertexShader);
    glAttachShader(programID,fragmentShader);
    glLinkProgram(programID);
    glGetProgramiv(programID,GL_LINK_STATUS,&success);
    if(!success)
    {
        glGetProgramInfoLog(programID,1024,nullptr,errorlog);
        std::cerr << "Failed to Compile Fragment shader: " << errorlog;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

}

void Shader::Use()
{
    glUseProgram(programID);
}

void Shader::SetFloatParam(const std::string &paramName, float val){

    glUniform1f(glGetUniformLocation(programID,paramName.c_str()),val);
}

void Shader::SetVec3Param(const std::string &paramName,glm::vec3 val){

    glUniform3f(glGetUniformLocation(programID,paramName.c_str()),val.x,val.y,val.z);

}

void Shader::SetMat4Param(const std::string &paramName,glm::mat4 val){

    glUniformMatrix4fv(glGetUniformLocation(programID,paramName.c_str()),1, GL_FALSE, glm::value_ptr(val));

}
