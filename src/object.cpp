#include <Object.h>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <iostream>


Object::Object(const Model* model,glm::vec3 position, glm::vec3 rotation, glm::vec3 scale)
    : model(model), position(position),rotation(rotation),scale(scale)
{
}

glm::mat4 Object::GetTransformationMatrix(){

    glm::mat4 t = glm::mat4(1.f);


    t = glm::translate(t,position);
    t = t * glm::mat4_cast(glm::quat(glm::radians(rotation)));
    t = glm::scale(t,scale);

    return t;

}

void Object::SetPosition(float x, float y, float z){

    // std::cout << "before:" << std::endl;
    // std::cout << position.x << " " << position.y << " " << position.z << " " << std::endl;   

    position.x = x;
    position.y = y;
    position.z = z;

    // std::cout << "after:" << std::endl;
    // std::cout << position.x << " " << position.y << " " << position.z << " \n" << std::endl;   
}


void Object::Draw(Shader & shader,glm::vec3 color){

    /** TODO: Is it possible to draw them without putting them into a matrix to save space? */
    shader.SetMat4Param("model",GetTransformationMatrix());
    shader.SetVec3Param("color",color);
    model->Draw(shader,GetTransformationMatrix());

}


