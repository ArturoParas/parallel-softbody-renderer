#version 330 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform float ambientStrength;
uniform vec3 lightColor;
uniform vec3 lightPos;

uniform vec3 color;

void main(){

    vec3 resultColor = vec3(0.0);

    vec3 Ia = ambientStrength * lightColor;

    vec3 N = normalize(Normal);
    vec3 Lm = normalize(lightPos - FragPos);
    vec3 Id = max(dot(N,Lm),0.0) * lightColor;


    resultColor = (Ia + Id) * color;
    FragColor = vec4(resultColor,1.0);
} 