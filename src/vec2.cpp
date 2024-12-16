#include "../include/vec2.hpp"
#include <bits/algorithmfwd.h>

using namespace softbody_sim;

Vec2::Vec2(const float c) : x_(c), y_(c) {};
Vec2::Vec2(const float x, const float y) : x_(x), y_(y) {};

void Vec2::set(const Vec2& vec)
{
  x_ = vec.x_;
  y_ = vec.y_;
}

float Vec2::get_x() const
{
  return x_;
}

float Vec2::get_y() const
{
  return y_;
}

void Vec2::vec_add(const Vec2& vec)
{
  x_ += vec.x_;
  y_ += vec.y_;
}

void Vec2::vec_sub(const Vec2& vec)
{
  x_ -= vec.x_;
  y_ -= vec.y_;
}

void Vec2::scalar_mul(const float scalar)
{
  x_ *= scalar;
  y_ *= scalar;
}

float Vec2::dot(const Vec2& vec)
{
  return x_ * vec.x_ + y_ * vec.y_;
}

bool Vec2::is_eq(const Vec2& vec)
{
  return x_ == vec.x_ && y_ == vec.y_;
}

void Vec2::clamp(const Vec2& min, const Vec2& max)
{
  x_ = std::min(std::max(x_, min.x_), max.x_);
  y_ = std::min(std::max(y_, min.y_), max.y_);
}
