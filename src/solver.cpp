#include <cmath>

#include "../include/solver.hpp"

using namespace softbody_sim;

Solver::Solver(const int width, const int height, std::vector<Circle>& circle,
  const int intermediate_steps, const float dt) : width_(width), height_(height),
  num_rows_(ceil(height / Circle::diameter) + 2), num_cols_(ceil(width / Circle::diameter) + 2),
  grid_(std::vector<std::vector<std::vector<Circle>>>(
    num_rows_, std::vector<std::vector<Circle>>(num_cols_, std::vector<Circle>(0)))),
  next_grid_(std::vector<std::vector<std::vector<Circle>>>(
    num_rows_, std::vector<std::vector<Circle>>(num_cols_, std::vector<Circle>(0)))),
  intermediate_steps_(intermediate_steps), dt_(dt), dt_intermediate_(dt / intermediate_steps),
  dt2_intermediate_(dt_intermediate_ * dt_intermediate_) {};

void Solver::insert_circle(Circle& circle)
{
  int row = int(circle.get_x() / Circle::diameter) + 1;
  int col = int(circle.get_y() / Circle::diameter) + 1;
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

void Solver::resolve_cell(const int row, const int col)
{
  std::vector<Circle> TL = grid_[row - 1][col - 1];
  std::vector<Circle> TM = grid_[row - 1][col];
  std::vector<Circle> TR = grid_[row - 1][col + 1];
  std::vector<Circle> ML = grid_[row][col - 1];
  std::vector<Circle> MM = grid_[row][col];
  std::vector<Circle> MR = grid_[row][col + 1];
  std::vector<Circle> BL = grid_[row + 1][col - 1];
  std::vector<Circle> BM = grid_[row + 1][col];
  std::vector<Circle> BR = grid_[row + 1][col + 1];
  for (auto& curr_circle : MM) {
    for (auto& comp_circle : TL) {
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : TM) {
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : TR) {
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : ML) {
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : MM) {
      if (!curr_circle.is_eq(comp_circle)) {
        curr_circle.resolve_collision(comp_circle);
      }
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : MR) {
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : BL) {
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : BM) {
      curr_circle.resolve_collision(comp_circle);
    }
    for (auto& comp_circle : BR) {
      curr_circle.resolve_collision(comp_circle);
    }
  }
}

void Solver::resolve_collisions()
{
  for (int row = 1; row < num_rows_ - 1; row++) {
    for (int col = 1; col < num_cols_ - 1; col++) {
      resolve_cell(row, col);
    }
  }
}

void Solver::apply_border()
{
  Vec2 min(Circle::rad, Circle::rad);
  Vec2 max(width_ - Circle::rad, height_ - Circle::rad);
  for (auto& row : grid_) {
    for (auto& cell : row) {
      for (auto& circle : cell) {
        circle.clamp_p_temp(min, max);
      }
    }
  }
}

void Solver::transition_circles()
{
  for (auto& row : grid_) {
    for (auto& cell : row) {
      int num_circles = cell.size();
      for (int i = 0; i < num_circles; i++) {
        Circle circle = cell.back();
        cell.pop_back();
        circle.update_pos_resolved();
        insert_circle(circle);
      }
    }
  }
}

void Solver::update()
{
  for (int i = 0; i < intermediate_steps_; i++) {
    move_circles();
    resolve_collisions();
    apply_border();
    transition_circles();
    transition_grid();
  }
}
