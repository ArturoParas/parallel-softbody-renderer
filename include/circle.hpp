#ifndef CIRCLE_HPP
#define CIRCLE_HPP

#include "vec2.hpp"

namespace softbody_sim
{

class Circle
{
public:
  constexpr static float rad = 2.5;
  constexpr static float diameter = rad * 2;
  constexpr static float mass = 1;
  constexpr static float g = 100;
  // static float v_thresh;

  Circle();
  Circle(const Vec2& p_prev, const Vec2& p_curr);

  void update_pos(const float dt2);

  void resolve_collision(const Circle& circle);

  void update_pos_resolved();

  Vec2 get_accel() const;
  Vec2 get_pos() const;
  Vec2 get_temp_pos() const;
  float get_x() const;
  float get_y() const;

  bool is_eq(const Circle& circle);

  void clamp_p_temp(const Vec2& min, const Vec2& max);

private:
  Vec2 p_curr_;
  Vec2 p_prev_;
  Vec2 p_temp_;
};

} // namespace softbody_sim

#endif // CIRCLE_HPP
