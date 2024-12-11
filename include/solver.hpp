#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>
#include <cmath>

#include "circle.hpp"

namespace softbody_sim
{

  

struct SolverInfo{

  int width;
  int height;
  int num_rows;
  int num_cols;

  int num_circles;
  float circle_radius;

  int num_springs;
  float k_constant;


  int intermediate_steps;
  float dt;
  float dt_intermediate;
  float dt2_intermediate;





  SolverInfo(int _width, int _height, int _num_circles,float _circle_radius, int _num_springs, float _k_constant,  int _intermediate_steps=8, float _dt=0.1f) 
    : width(_width), height(_height), 
      num_rows(ceil(height / 2 * _circle_radius) + 2), num_cols(ceil(width / 2 * _circle_radius) + 2),
      num_circles(_num_circles), circle_radius(_circle_radius), num_springs(_num_springs), k_constant(_k_constant),
      intermediate_steps(_intermediate_steps), dt(_dt), 
      dt_intermediate(_dt/_intermediate_steps), dt2_intermediate(dt_intermediate*dt_intermediate) {}


};




class Solver
{
public:
  Solver(const int width, const int height, std::vector<Circle>& circles,
    const int intermediate_steps=8, const float dt=0.1);

  void insert_circle(Circle& circle);

  void transition_grid();

  void move_circles();

  void resolve_cell(const int row, const int col);

  void resolve_collisions();

  void apply_border();

  void transition_circles();

  void update();

  void print_grid();

  void print_next_grid();

  float get_dt() const;

private:
  int width_;
  int height_;
  int num_cols_;
  int num_rows_;
  std::vector<std::vector<std::vector<Circle>>> grid_;
  std::vector<std::vector<std::vector<Circle>>> next_grid_;
  int intermediate_steps_;
  float dt_;
  float dt_intermediate_;
  float dt2_intermediate_;
};

} // namespace softbody_sim

#endif // SOLVER_HPP