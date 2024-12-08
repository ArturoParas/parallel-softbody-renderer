#include <cmath>

#include "../include/circle.hpp"

using namespace softbody_sim;

Circle::Circle(const Vec2& p_prev, const Vec2& p_curr)
  : p_prev_(p_prev), p_curr_(p_curr), p_res_(p_curr) {};

Vec2 Circle::get_accel()
{
  return Vec2(0, Circle::g);
}

void Circle::update_pos(const float dt2)
{
  Vec2 accel = get_accel();
  Vec2 p_next(p_curr_);
  /** TODO: If using v_thresh, then compute vel and clamp according to v_thresh */
  p_next.vec_add(p_curr_);
  p_next.vec_sub(p_prev_);
  accel.scalar_mul(dt2);
  p_next.vec_add(accel);

  p_prev_.set(p_curr_);
  p_curr_.set(p_next);
  p_res_.set(p_next);
}

void Circle::resolve_collision(const Circle& circle)
{
  Vec2 p_diff(p_curr_);
  p_diff.vec_sub(circle.p_curr_);
  float dist = sqrt(p_diff.dot(p_diff));
  float overlap = dist - Circle::diameter;
  if (overlap < 0) {
    p_diff.scalar_mul(overlap * 0.5 / dist);
    p_res_.vec_add(p_diff);
  }
}

void Circle::update_pos_resolved()
{
  p_curr_.set(p_res_);
}

float Circle::x() const
{
  return p_curr_.x();
}

float Circle::x() const
{
  return p_curr_.y();
}
