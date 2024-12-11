#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <vector>
#include <unordered_map>

struct Pt3
{
  Pt3(const float x_, const float y_, const float z_);
  float x;
  float y;
  float z;
  bool operator==(const Pt3& other) const;
};

struct Spring
{
  Spring(const std::size_t idx1_, const std::size_t idx2_);
  std::size_t idx1;
  std::size_t idx2;
};

template <>
struct std::hash<Pt3>
{
  std::size_t operator()(const Pt3& pt) const;
};

void get_sphere_pts(
  const int rad, const Pt3& center, const int rest_len, std::vector<Pt3>& pts,
  std::unordered_map<Pt3, std::size_t>& pt_to_idx);

void get_springs(
  const int rest_len, std::unordered_map<Pt3, std::size_t>& pt_to_idx,
  std::vector<Spring>& springs);

void print_sphere_stats(const std::vector<Pt3>& pts, const std::vector<Spring>& springs);

#endif // PREPROCESSING_HPP
