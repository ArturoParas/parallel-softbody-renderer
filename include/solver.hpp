#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>
#include <cmath>

#include "circle.hpp"

namespace softbody_sim
{

struct SolverInfo{

  float width; //x
  float depth; //y
  float height; //z

  float circle_radius;
  float circle_mass;

  float k_constant;
  float spring_rest_length;

  float damping_constant;
  float gravity_acc;

  uint32_t intermediate_steps;
  float dt2_intermediate;

  uint32_t num_blocks;
  uint32_t threads_per_block;

  int max_pts;
  int max_nbors_per_pt;
  int max_nbors_per_block;
  int max_nbors;
  int max_rdonly_per_block;
  int max_rdonly;

  SolverInfo(uint32_t _width=1000, uint32_t _depth=1000, uint32_t _height=1000, 
             float _circle_radius=0.25f, float _circle_mass = 1.f,float _k_constant=2.f, float _spring_rest_length=1.f,
             float _damping_constant=0.98f, float _gravity_acc=-64.f,
             uint32_t _intermediate_steps=8, float _dt=0.1f, 
             uint32_t _num_blocks=2, uint32_t _threads_per_block=4) 

    : width(_width), depth(_depth), height(_height), 
      circle_radius(_circle_radius), circle_mass(_circle_mass), k_constant(_k_constant), spring_rest_length(_spring_rest_length), 
      damping_constant(_damping_constant), gravity_acc(_gravity_acc),
      intermediate_steps(_intermediate_steps), dt2_intermediate((_dt*_dt)/(_intermediate_steps*_intermediate_steps)), 
      num_blocks(_num_blocks), threads_per_block(_threads_per_block) {}


};

} 

#endif // SOLVER_HPP