#include <Model.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <iostream>

//Probably will move these to another file 

static const std::vector<glm::vec3> MODEL_PRIMITIVE_CUBE_VERTICES = 
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

static const std::vector<uint32_t> MODEL_PRIMITIVE_CUBE_INDICES = 
{
    0,1,2,2,3,0,
    4,5,6,6,7,4,
    1,5,6,6,7,4,
    0,4,7,7,3,0,
    3,2,6,6,7,3,
    0,1,5,5,4,0
};

static const std::vector<glm::vec3> MODEL_PRIMITIVE_ICOSPHERE2_VERTICES = 
{
    glm::vec3(-0.5257f, 0.8507f, 0.0f),glm::vec3(0.5257f, 0.8507f, 0.0f),glm::vec3(-0.5257f, -0.8507f, 0.0f),glm::vec3(0.5257f, -0.8507f, 0.0f),glm::vec3(0.0f, -0.5257f, 0.8507f),glm::vec3(0.0f, 0.5257f, 0.8507f),glm::vec3(0.0f, -0.5257f, -0.8507f),glm::vec3(0.0f, 0.5257f, -0.8507f),glm::vec3(0.8507f, 0.0f, -0.5257f),glm::vec3(0.8507f, 0.0f, 0.5257f),glm::vec3(-0.8507f, 0.0f, -0.5257f),glm::vec3(-0.8507f, 0.0f, 0.5257f),glm::vec3(-0.809f, 0.5f, 0.309f),glm::vec3(-0.5f, 0.309f, 0.809f),glm::vec3(-0.309f, 0.809f, 0.5f),glm::vec3(0.309f, 0.809f, 0.5f),glm::vec3(0.0f, 1.0f, 0.0f),glm::vec3(0.309f, 0.809f, -0.5f),glm::vec3(-0.309f, 0.809f, -0.5f),glm::vec3(-0.5f, 0.309f, -0.809f),glm::vec3(-0.809f, 0.5f, -0.309f),glm::vec3(-1.0f, 0.0f, 0.0f),glm::vec3(0.5f, 0.309f, 0.809f),glm::vec3(0.809f, 0.5f, 0.309f),glm::vec3(-0.5f, -0.309f, 0.809f),glm::vec3(0.0f, 0.0f, 1.0f),glm::vec3(-0.809f, -0.5f, -0.309f),glm::vec3(-0.809f, -0.5f, 0.309f),glm::vec3(0.0f, 0.0f, -1.0f),glm::vec3(-0.5f, -0.309f, -0.809f),glm::vec3(0.809f, 0.5f, -0.309f),glm::vec3(0.5f, 0.309f, -0.809f),glm::vec3(0.809f, -0.5f, 0.309f),glm::vec3(0.5f, -0.309f, 0.809f),glm::vec3(0.309f, -0.809f, 0.5f),glm::vec3(-0.309f, -0.809f, 0.5f),glm::vec3(0.0f, -1.0f, 0.0f),glm::vec3(-0.309f, -0.809f, -0.5f),glm::vec3(0.309f, -0.809f, -0.5f),glm::vec3(0.5f, -0.309f, -0.809f),glm::vec3(0.809f, -0.5f, -0.309f),glm::vec3(1.0f, 0.0f, 0.0f),glm::vec3(-0.6938f, 0.702f, 0.1606f),glm::vec3(-0.5878f, 0.6882f, 0.4253f),glm::vec3(-0.4339f, 0.8627f, 0.2599f),glm::vec3(-0.702f, 0.1606f, 0.6938f),glm::vec3(-0.6882f, 0.4253f, 0.5878f),glm::vec3(-0.8627f, 0.2599f, 0.4339f),glm::vec3(-0.1606f, 0.6938f, 0.702f),glm::vec3(-0.4253f, 0.5878f, 0.6882f),glm::vec3(-0.2599f, 0.4339f, 0.8627f),glm::vec3(-0.1625f, 0.9511f, 0.2629f),glm::vec3(-0.2733f, 0.9619f, 0.0f),glm::vec3(0.1606f, 0.6938f, 0.702f),glm::vec3(0.0f, 0.8507f, 0.5257f),glm::vec3(0.2733f, 0.9619f, 0.0f),glm::vec3(0.1625f, 0.9511f, 0.2629f),glm::vec3(0.4339f, 0.8627f, 0.2599f),glm::vec3(-0.1625f, 0.9511f, -0.2629f),glm::vec3(-0.4339f, 0.8627f, -0.2599f),glm::vec3(0.4339f, 0.8627f, -0.2599f),glm::vec3(0.1625f, 0.9511f, -0.2629f),glm::vec3(-0.1606f, 0.6938f, -0.702f),glm::vec3(0.0f, 0.8507f, -0.5257f),glm::vec3(0.1606f, 0.6938f, -0.702f),glm::vec3(-0.5878f, 0.6882f, -0.4253f),glm::vec3(-0.6938f, 0.702f, -0.1606f),glm::vec3(-0.2599f, 0.4339f, -0.8627f),glm::vec3(-0.4253f, 0.5878f, -0.6882f),glm::vec3(-0.8627f, 0.2599f, -0.4339f),glm::vec3(-0.6882f, 0.4253f, -0.5878f),glm::vec3(-0.702f, 0.1606f, -0.6938f),glm::vec3(-0.8507f, 0.5257f, 0.0f),glm::vec3(-0.9619f, 0.0f, -0.2733f),glm::vec3(-0.9511f, 0.2629f, -0.1625f),glm::vec3(-0.9511f, 0.2629f, 0.1625f),glm::vec3(-0.9619f, 0.0f, 0.2733f),glm::vec3(0.5878f, 0.6882f, 0.4253f),glm::vec3(0.6938f, 0.702f, 0.1606f),glm::vec3(0.2599f, 0.4339f, 0.8627f),glm::vec3(0.4253f, 0.5878f, 0.6882f),glm::vec3(0.8627f, 0.2599f, 0.4339f),glm::vec3(0.6882f, 0.4253f, 0.5878f),glm::vec3(0.702f, 0.1606f, 0.6938f),glm::vec3(-0.2629f, 0.1625f, 0.9511f),glm::vec3(0.0f, 0.2733f, 0.9619f),glm::vec3(-0.702f, -0.1606f, 0.6938f),glm::vec3(-0.5257f, 0.0f, 0.8507f),glm::vec3(0.0f, -0.2733f, 0.9619f),glm::vec3(-0.2629f, -0.1625f, 0.9511f),glm::vec3(-0.2599f, -0.4339f, 0.8627f),glm::vec3(-0.9511f, -0.2629f, 0.1625f),glm::vec3(-0.8627f, -0.2599f, 0.4339f),glm::vec3(-0.8627f, -0.2599f, -0.4339f),glm::vec3(-0.9511f, -0.2629f, -0.1625f),glm::vec3(-0.6938f, -0.702f, 0.1606f),glm::vec3(-0.8507f, -0.5257f, 0.0f),glm::vec3(-0.6938f, -0.702f, -0.1606f),glm::vec3(-0.5257f, 0.0f, -0.8507f),glm::vec3(-0.702f, -0.1606f, -0.6938f),glm::vec3(0.0f, 0.2733f, -0.9619f),glm::vec3(-0.2629f, 0.1625f, -0.9511f),glm::vec3(-0.2599f, -0.4339f, -0.8627f),glm::vec3(-0.2629f, -0.1625f, -0.9511f),glm::vec3(0.0f, -0.2733f, -0.9619f),glm::vec3(0.4253f, 0.5878f, -0.6882f),glm::vec3(0.2599f, 0.4339f, -0.8627f),glm::vec3(0.6938f, 0.702f, -0.1606f),glm::vec3(0.5878f, 0.6882f, -0.4253f),glm::vec3(0.702f, 0.1606f, -0.6938f),glm::vec3(0.6882f, 0.4253f, -0.5878f),glm::vec3(0.8627f, 0.2599f, -0.4339f),glm::vec3(0.6938f, -0.702f, 0.1606f),glm::vec3(0.5878f, -0.6882f, 0.4253f),glm::vec3(0.4339f, -0.8627f, 0.2599f),glm::vec3(0.702f, -0.1606f, 0.6938f),glm::vec3(0.6882f, -0.4253f, 0.5878f),glm::vec3(0.8627f, -0.2599f, 0.4339f),glm::vec3(0.1606f, -0.6938f, 0.702f),glm::vec3(0.4253f, -0.5878f, 0.6882f),glm::vec3(0.2599f, -0.4339f, 0.8627f),glm::vec3(0.1625f, -0.9511f, 0.2629f),glm::vec3(0.2733f, -0.9619f, 0.0f),glm::vec3(-0.1606f, -0.6938f, 0.702f),glm::vec3(0.0f, -0.8507f, 0.5257f),glm::vec3(-0.2733f, -0.9619f, 0.0f),glm::vec3(-0.1625f, -0.9511f, 0.2629f),glm::vec3(-0.4339f, -0.8627f, 0.2599f),glm::vec3(0.1625f, -0.9511f, -0.2629f),glm::vec3(0.4339f, -0.8627f, -0.2599f),glm::vec3(-0.4339f, -0.8627f, -0.2599f),glm::vec3(-0.1625f, -0.9511f, -0.2629f),glm::vec3(0.1606f, -0.6938f, -0.702f),glm::vec3(0.0f, -0.8507f, -0.5257f),glm::vec3(-0.1606f, -0.6938f, -0.702f),glm::vec3(0.5878f, -0.6882f, -0.4253f),glm::vec3(0.6938f, -0.702f, -0.1606f),glm::vec3(0.2599f, -0.4339f, -0.8627f),glm::vec3(0.4253f, -0.5878f, -0.6882f),glm::vec3(0.8627f, -0.2599f, -0.4339f),glm::vec3(0.6882f, -0.4253f, -0.5878f),glm::vec3(0.702f, -0.1606f, -0.6938f),glm::vec3(0.8507f, -0.5257f, 0.0f),glm::vec3(0.9619f, 0.0f, -0.2733f),glm::vec3(0.9511f, -0.2629f, -0.1625f),glm::vec3(0.9511f, -0.2629f, 0.1625f),glm::vec3(0.9619f, 0.0f, 0.2733f),glm::vec3(0.2629f, -0.1625f, 0.9511f),glm::vec3(0.5257f, 0.0f, 0.8507f),glm::vec3(0.2629f, 0.1625f, 0.9511f),glm::vec3(-0.5878f, -0.6882f, 0.4253f),glm::vec3(-0.4253f, -0.5878f, 0.6882f),glm::vec3(-0.6882f, -0.4253f, 0.5878f),glm::vec3(-0.4253f, -0.5878f, -0.6882f),glm::vec3(-0.5878f, -0.6882f, -0.4253f),glm::vec3(-0.6882f, -0.4253f, -0.5878f),glm::vec3(0.5257f, 0.0f, -0.8507f),glm::vec3(0.2629f, -0.1625f, -0.9511f),glm::vec3(0.2629f, 0.1625f, -0.9511f),glm::vec3(0.9511f, 0.2629f, 0.1625f),glm::vec3(0.9511f, 0.2629f, -0.1625f),glm::vec3(0.8507f, 0.5257f, 0.0f)
};

static const std::vector<uint32_t> MODEL_PRIMITIVE_ICOSPHERE2_INDICES = 
{
    0, 42, 44, 12, 43, 42, 14, 44, 43, 42, 43, 44, 11, 45, 47, 13, 46, 45, 12, 47, 46, 45, 46, 47, 5, 48, 50, 14, 49, 48, 13, 50, 49, 48, 49, 50, 12, 46, 43, 13, 49, 46, 14, 43, 49, 46, 49, 43, 0, 44, 52, 14, 51, 44, 16, 52, 51, 44, 51, 52, 5, 53, 48, 15, 54, 53, 14, 48, 54, 53, 54, 48, 1, 55, 57, 16, 56, 55, 15, 57, 56, 55, 56, 57, 14, 54, 51, 15, 56, 54, 16, 51, 56, 54, 56, 51, 0, 52, 59, 16, 58, 52, 18, 59, 58, 52, 58, 59, 1, 60, 55, 17, 61, 60, 16, 55, 61, 60, 61, 55, 7, 62, 64, 18, 63, 62, 17, 64, 63, 62, 63, 64, 16, 61, 58, 17, 63, 61, 18, 58, 63, 61, 63, 58, 0, 59, 66, 18, 65, 59, 20, 66, 65, 59, 65, 66, 7, 67, 62, 19, 68, 67, 18, 62, 68, 67, 68, 62, 10, 69, 71, 20, 70, 69, 19, 71, 70, 69, 70, 71, 18, 68, 65, 19, 70, 68, 20, 65, 70, 68, 70, 65, 0, 66, 42, 20, 72, 66, 12, 42, 72, 66, 72, 42, 10, 73, 69, 21, 74, 73, 20, 69, 74, 73, 74, 69, 11, 47, 76, 12, 75, 47, 21, 76, 75, 47, 75, 76, 20, 74, 72, 21, 75, 74, 12, 72, 75, 74, 75, 72, 1, 57, 78, 15, 77, 57, 23, 78, 77, 57, 77, 78, 5, 79, 53, 22, 80, 79, 15, 53, 80, 79, 80, 53, 9, 81, 83, 23, 82, 81, 22, 83, 82, 81, 82, 83, 15, 80, 77, 22, 82, 80, 23, 77, 82, 80, 82, 77, 5, 50, 85, 13, 84, 50, 25, 85, 84, 50, 84, 85, 11, 86, 45, 24, 87, 86, 13, 45, 87, 86, 87, 45, 4, 88, 90, 25, 89, 88, 24, 90, 89, 88, 89, 90, 13, 87, 84, 24, 89, 87, 25, 84, 89, 87, 89, 84, 11, 76, 92, 21, 91, 76, 27, 92, 91, 76, 91, 92, 10, 93, 73, 26, 94, 93, 21, 73, 94, 93, 94, 73, 2, 95, 97, 27, 96, 95, 26, 97, 96, 95, 96, 97, 21, 94, 91, 26, 96, 94, 27, 91, 96, 94, 96, 91, 10, 71, 99, 19, 98, 71, 29, 99, 98, 71, 98, 99, 7, 100, 67, 28, 101, 100, 19, 67, 101, 100, 101, 67, 6, 102, 104, 29, 103, 102, 28, 104, 103, 102, 103, 104, 19, 101, 98, 28, 103, 101, 29, 98, 103, 101, 103, 98, 7, 64, 106, 17, 105, 64, 31, 106, 105, 64, 105, 106, 1, 107, 60, 30, 108, 107, 17, 60, 108, 107, 108, 60, 8, 109, 111, 31, 110, 109, 30, 111, 110, 109, 110, 111, 17, 108, 105, 30, 110, 108, 31, 105, 110, 108, 110, 105, 3, 112, 114, 32, 113, 112, 34, 114, 113, 112, 113, 114, 9, 115, 117, 33, 116, 115, 32, 117, 116, 115, 116, 117, 4, 118, 120, 34, 119, 118, 33, 120, 119, 118, 119, 120, 32, 116, 113, 33, 119, 116, 34, 113, 119, 116, 119, 113, 3, 114, 122, 34, 121, 114, 36, 122, 121, 114, 121, 122, 4, 123, 118, 35, 124, 123, 34, 118, 124, 123, 124, 118, 2, 125, 127, 36, 126, 125, 35, 127, 126, 125, 126, 127, 34, 124, 121, 35, 126, 124, 36, 121, 126, 124, 126, 121, 3, 122, 129, 36, 128, 122, 38, 129, 128, 122, 128, 129, 2, 130, 125, 37, 131, 130, 36, 125, 131, 130, 131, 125, 6, 132, 134, 38, 133, 132, 37, 134, 133, 132, 133, 134, 36, 131, 128, 37, 133, 131, 38, 128, 133, 131, 133, 128, 3, 129, 136, 38, 135, 129, 40, 136, 135, 129, 135, 136, 6, 137, 132, 39, 138, 137, 38, 132, 138, 137, 138, 132, 8, 139, 141, 40, 140, 139, 39, 141, 140, 139, 140, 141, 38, 138, 135, 39, 140, 138, 40, 135, 140, 138, 140, 135, 3, 136, 112, 40, 142, 136, 32, 112, 142, 136, 142, 112, 8, 143, 139, 41, 144, 143, 40, 139, 144, 143, 144, 139, 9, 117, 146, 32, 145, 117, 41, 146, 145, 117, 145, 146, 40, 144, 142, 41, 145, 144, 32, 142, 145, 144, 145, 142, 4, 120, 88, 33, 147, 120, 25, 88, 147, 120, 147, 88, 9, 83, 115, 22, 148, 83, 33, 115, 148, 83, 148, 115, 5, 85, 79, 25, 149, 85, 22, 79, 149, 85, 149, 79, 33, 148, 147, 22, 149, 148, 25, 147, 149, 148, 149, 147, 2, 127, 95, 35, 150, 127, 27, 95, 150, 127, 150, 95, 4, 90, 123, 24, 151, 90, 35, 123, 151, 90, 151, 123, 11, 92, 86, 27, 152, 92, 24, 86, 152, 92, 152, 86, 35, 151, 150, 24, 152, 151, 27, 150, 152, 151, 152, 150, 6, 134, 102, 37, 153, 134, 29, 102, 153, 134, 153, 102, 2, 97, 130, 26, 154, 97, 37, 130, 154, 97, 154, 130, 10, 99, 93, 29, 155, 99, 26, 93, 155, 99, 155, 93, 37, 154, 153, 26, 155, 154, 29, 153, 155, 154, 155, 153, 8, 141, 109, 39, 156, 141, 31, 109, 156, 141, 156, 109, 6, 104, 137, 28, 157, 104, 39, 137, 157, 104, 157, 137, 7, 106, 100, 31, 158, 106, 28, 100, 158, 106, 158, 100, 39, 157, 156, 28, 158, 157, 31, 156, 158, 157, 158, 156, 9, 146, 81, 41, 159, 146, 23, 81, 159, 146, 159, 81, 8, 111, 143, 30, 160, 111, 41, 143, 160, 111, 160, 143, 1, 78, 107, 23, 161, 78, 30, 107, 161, 78, 161, 107, 41, 160, 159, 30, 161, 160, 23, 159, 161, 160, 161, 159
};

Model::Model(const std::string & filename)
{

    Assimp::Importer importer;
    importer.ReadFile(filename,aiProcess_Triangulate);

    const aiScene* scene = importer.GetScene();

    if(!scene)
    {
        std::cerr << "Failed to load model (Scene null): " << filename 
                  << " emsg: "  << importer.GetErrorString() << "\n";
        return;
    }
    if(!scene->mRootNode)
    {
        std::cerr << "Failed to load model (Root null): " << filename 
                  << " emsg: "  << importer.GetErrorString() << "\n";
        return;
    }
    if(scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
    {
        std::cerr << "Failed to load model (Incomplete): " << filename 
                  << " emsg: "  << importer.GetErrorString() << "\n";
        return;
    }


    ProcessNode(scene->mRootNode, scene, glm::mat4(1.f));

}
Model::Model(const int primitiveID)
{

    std::vector<glm::vec3> src_vertices;
    std::vector<uint32_t> src_indices;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;


    //Defaults to unit cube
    switch(primitiveID)
    {

        case MODEL_PRIMITIVE_ICOSPHERE2:
            src_vertices = MODEL_PRIMITIVE_ICOSPHERE2_VERTICES;
            src_indices = MODEL_PRIMITIVE_ICOSPHERE2_INDICES;
            break;

        default:
            src_vertices = MODEL_PRIMITIVE_CUBE_VERTICES;
            src_indices = MODEL_PRIMITIVE_CUBE_INDICES;
            break;

    } 

    for(int vi = 0; vi < src_vertices.size();vi++)
    {

        Vertex vert;
        vert.pos = glm::vec3(src_vertices[vi].x, src_vertices[vi].y, src_vertices[vi].z);

        vertices.push_back(vert);
    }

    for(int ii = 0; ii < src_indices.size(); ii++)
    {
        indices.push_back(src_indices[ii]);
    }

    Mesh mesh(vertices,indices);
    meshes.push_back(mesh);

}


void Model::Draw(Shader & shader, glm::mat4 objectTransformation) const
{

    for(const auto & mesh : meshes)
    {

        shader.Use();
        shader.SetMat4Param("model",mesh.transformation * objectTransformation);
        mesh.Draw();
    }


}

Mesh Model::ProcessMesh(aiMesh* mesh){

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    for(uint32_t i = 0; i < mesh->mNumVertices; i++)
    {
        aiVector3D position_vec = mesh->mVertices[i];
        aiVector3D normal_vec = mesh->mNormals[i];

        Vertex vert;
        vert.pos = glm::vec3(position_vec.x, position_vec.y, position_vec.z);
        vert.norm = glm::vec3(normal_vec.x, normal_vec.y, normal_vec.z);

        vertices.push_back(vert);

    }

    for(uint32_t i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];

        for(uint32_t j = 0; j < face.mNumIndices; j++)
        {
            indices.push_back(face.mIndices[j]);
        }

    }

    return Mesh(vertices,indices);
}

void Model::ProcessNode(aiNode* node, const aiScene* scene, glm::mat4 parentTransformation){

    std::cerr << node->mName.C_Str() << "\n";

    glm::mat4 transformation;

    transformation[0][0] = node->mTransformation.a1;
    transformation[1][0] = node->mTransformation.a2;
    transformation[2][0] = node->mTransformation.a3;
    transformation[3][0] = node->mTransformation.a4;
    transformation[0][1] = node->mTransformation.b1;
    transformation[1][1] = node->mTransformation.b2;
    transformation[2][1] = node->mTransformation.b3;
    transformation[3][1] = node->mTransformation.b4;
    transformation[0][2] = node->mTransformation.c1;
    transformation[1][2] = node->mTransformation.c2;
    transformation[2][2] = node->mTransformation.c3;
    transformation[3][2] = node->mTransformation.c4;
    transformation[0][3] = node->mTransformation.d1;
    transformation[1][3] = node->mTransformation.d2;
    transformation[2][3] = node->mTransformation.d3;
    transformation[3][3] = node->mTransformation.d4;

    transformation = parentTransformation * transformation;

    std::cerr << transformation[0][0] << "  " << transformation[0][1] << "  " <<  transformation[0][2] << "  "  << transformation[0][3] << "\n" 
              << transformation[1][0] << "  " << transformation[1][1] << "  " <<  transformation[1][2] << "  "  << transformation[1][3] << "\n" 
              << transformation[2][0] << "  " << transformation[2][1] << "  " <<  transformation[2][2] << "  "  << transformation[2][3] << "\n" 
              << transformation[3][0] << "  " << transformation[3][1] << "  " <<  transformation[3][2] << "  "  << transformation[3][3] << "\n\n";


    for(uint32_t i = 0; i < node->mNumMeshes; i++)
    {
        Mesh mesh = ProcessMesh(scene->mMeshes[node->mMeshes[i]]);
        mesh.transformation = transformation;
        meshes.push_back(mesh);
    }

    for(uint32_t i = 0; i < node->mNumChildren; i++)
    {
        ProcessNode(node->mChildren[i],scene,transformation);
    }

}