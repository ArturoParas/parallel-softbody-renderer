#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>

#include "circle.hpp"

namespace softbody_sim
{

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
  int num_rows_;
  int num_cols_;
  std::vector<std::vector<std::vector<Circle>>> grid_;
  std::vector<std::vector<std::vector<Circle>>> next_grid_;
  int intermediate_steps_;
  float dt_;
  float dt_intermediate_;
  float dt2_intermediate_;
};

} // namespace softbody_sim

#endif // SOLVER_HPP
