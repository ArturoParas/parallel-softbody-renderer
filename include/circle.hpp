#ifndef CIRCLE_HPP
#define CIRCLE_HPP

#include "vec2.hpp"

namespace softbody_sim
{

class Circle
{
public:
  constexpr static float rad = 25;
  constexpr static float diameter = rad * 2;
  constexpr static float mass = 1;
  constexpr static float g = 1000;
  // static float v_thresh;

  Circle(const Vec2& p_prev, const Vec2& p_curr);

  Vec2 get_accel();

  void update_pos(const float dt2);

  void resolve_collision(const Circle& circle);

  void update_pos_resolved();

  float x() const;
  float y() const;

private:
  Vec2 p_curr_;
  Vec2 p_prev_;
  Vec2 p_res_;
};

} // namespace softbody_sim

#endif // CIRCLE_HPP
