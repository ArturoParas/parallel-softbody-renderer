#ifndef VEC2_HPP
#define VEC2_HPP

namespace softbody_sim
{

class Vec2
{
public:
  Vec2(const float x, const float y);

  void set(const Vec2& vec);
  void set(const float x, const float y);

  float get_x() const;
  float get_y() const;

  bool is_eq(const Vec2& vec);

  void vec_add(const Vec2& vec);

  void vec_sub(const Vec2& vec);

  void scalar_mul(const float scalar);

  float dot(const Vec2& vec);

  void clamp(const Vec2& min, const Vec2& max);

private:
  float x_;
  float y_;
};

} // namespace softbody_sim

#endif // VEC2_HPP
