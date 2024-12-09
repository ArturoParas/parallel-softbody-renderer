#include <Mesh.h>

#include <GL/glew.h>

#include <iostream>

static const std::vector<glm::vec3> MESH_CUBE_VERTICES = 
{
    glm::vec3(-0.5f,-0.5f,-0.5f),
    glm::vec3(0.5f,-0.5f,-0.5f),
    glm::vec3(0.5f,0.5f,-0.5f),
    glm::vec3(-0.5f,0.5f,-0.5f),
    glm::vec3(-0.5f,-0.5f,0.5f),
    glm::vec3(0.5f,-0.5f,0.5f),
    glm::vec3(0.5f,0.5f,0.5f),
    glm::vec3(-0.5f,0.5f,0.5f)
};

static const std::vector<uint32_t> MESH_CUBE_INDICES = 
{
    0,1,2,2,3,0,
    4,5,6,6,7,4,
    1,5,6,6,7,4,
    0,4,7,7,3,0,
    3,2,6,6,7,3,
    0,1,5,5,4,0
};

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices)
   : vertices(vertices), indices(indices), vao(),vbo(),ebo(),transformation(glm::mat4(1.f))
{

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glGenBuffers(1,&ebo);
    glGenVertexArrays(1,&vao);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,vertices.size() * sizeof(Vertex),vertices.data(),GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.size() * sizeof(glm::uint32_t),indices.data(),GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,pos));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,norm));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
    glBindVertexArray(0);

}

void Mesh::Draw() const
{
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
    glDrawElements(GL_TRIANGLES,indices.size(),GL_UNSIGNED_INT,nullptr);

}