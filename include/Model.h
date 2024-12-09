#pragma once

#include <Mesh.h>
#include <Shader.h>

#include <assimp/scene.h>
#include <glm/glm.hpp>

#include <string>

class Model
{

public:

    static const int MODEL_PRIMITIVE_CUBE = 1;
    static const int MODEL_PRIMITIVE_ICOSPHERE2 = 2;

    Model(const std::string & filename);
    Model(const int primitiveID);
    void Draw(Shader & shader, glm::mat4 objectTransformation) const;


private:

    Mesh ProcessMesh(aiMesh* mesh);
    void ProcessNode(aiNode* node, const aiScene* scene,glm::mat4 parentTransformation);

    std::vector<Mesh> meshes;

};