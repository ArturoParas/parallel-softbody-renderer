#ifndef CREATE_INPUT_HPP
#define CREATE_INPUT_HPP

#include <vector>
#include <unordered_map>
#include <string>

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

void get_adjacency_list(
  const int rest_len, const std::unordered_map<Pt3, std::size_t>& pt_to_idx,
  std::vector<std::vector<std::size_t>>& adjacency_list);

void print_sphere_stats(const std::vector<Pt3>& pts, const std::vector<Spring>& springs);

void write_to_file(
  const std::vector<Pt3>& pts, const std::vector<std::vector<std::size_t>>& adjacency_list,
  const std::string file);

#endif // CREATE_INPUT_HPP
