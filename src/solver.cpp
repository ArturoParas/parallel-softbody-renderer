#include <cmath>

#include "../include/solver.hpp"

using namespace softbody_sim;

Solver::Solver(const int width, const int height, std::vector<Circle>& circle,
  const int intermediate_steps=8, const float dt=0.1) : width_(width), height_(height),
  num_rows_(ceil(height / Circle::diameter) + 2), num_cols_(ceil(width / Circle::diameter) + 2),
  grid_(std::vector<std::vector<std::vector<Circle>>>(
    num_rows_, std::vector<std::vector<Circle>>(num_cols_, std::vector<Circle>(0)))),
  next_grid_(std::vector<std::vector<std::vector<Circle>>>(
    num_rows_, std::vector<std::vector<Circle>>(num_cols_, std::vector<Circle>(0)))),
  intermediate_steps_(intermediate_steps), dt_(dt), dt_intermediate_(dt / intermediate_steps),
  dt2_intermediate_(dt_intermediate_ * dt_intermediate_) {};

void Solver::insert_circle(Circle& circle)
{
  int row = int(circle.x() / Circle::diameter) + 1;
  int col = int(circle.y() / Circle::diameter) + 1;
  next_grid_[row][col].push_back(circle);
}

void Solver::transition_grid()
{
  grid_ = next_grid_;
  next_grid_ = std::vector<std::vector<std::vector<Circle>>>(
    num_rows_, std::vector<std::vector<Circle>>(num_cols_, std::vector<Circle>(0)));
}

void Solver::move_circles()
{
  for (auto& row : grid_) {
    for (auto& cell : row) {
      for (auto& circle : cell) {
        circle.update_pos(dt2_intermediate_);
      }
    }
  }
}
