#ifndef CREATE_INPUT_HPP
#define CREATE_INPUT_HPP

#include <vector>
#include <unordered_map>
#include <string>

struct Pt3
{
  Pt3(const float x_, const float y_, const float z_);
  Pt3();
  float x;
  float y;
  float z;
  bool operator==(const Pt3& other) const;
};

template <>
struct std::hash<Pt3>
{
  std::size_t operator()(const Pt3& pt) const;
};

std::vector<int> block_threads_to_dims(const int block_threads, const int rest_len);

void get_sphere_pts(
  const int block_threads, const int rad, const Pt3& center, const int rest_len,
  std::vector<std::vector<Pt3>>& pts, std::unordered_map<Pt3, int>& pt_idxs);

void get_adjacency_list(
  const int rest_len, const std::vector<Pt3>& pts, const std::unordered_map<Pt3, int>& pt_idxs,
  std::vector<std::vector<int>>& adj_list);

void get_rd_only_idxs(
  const int block_threads, const std::vector<std::vector<Pt3>>& pts,
  const std::vector<std::vector<int>>& adj_list, std::vector<std::vector<int>>& rd_only_idxs);

void get_nbors_bufs(
  const int block_threads, const std::vector<std::vector<Pt3>>& pts,
  const std::vector<std::vector<int>>& adj_list, const std::vector<std::vector<int>>& rd_only_idxs,
  std::vector<std::vector<int>>& nbors_bufs);

void write_to_file(
  const int rad, const int rest_len, const std::vector<std::vector<Pt3>>& pts,
  const std::vector<std::vector<int>>& rd_only_idxs,
  const std::vector<std::vector<int>>& nbors_bufs, const std::string file);

void print_sphere_stats(
  const std::vector<std::vector<Pt3>>& pts, const std::vector<std::vector<int>>& adj_list,
  const std::vector<std::vector<int>>& rd_only_idxs,
  const std::vector<std::vector<int>>& nbors_bufs);

#endif // CREATE_INPUT_HPP
