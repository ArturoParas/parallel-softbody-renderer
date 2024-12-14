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

int get_pts(
  const int block_threads, const int rad, const Pt3& center, const int rest_len,
  std::vector<std::vector<Pt3>>& pts);

void reget_pts(
  const int num_blocks, const int max_pts, const std::vector<std::vector<Pt3>>& pts,
  std::vector<Pt3>& new_pts, std::unordered_map<Pt3, int>& map, std::vector<bool>& indicator);

int get_adj_list(
  const int rest_len, const std::vector<Pt3>& pts, const std::unordered_map<Pt3, int>& idx_map,
  const std::vector<bool>& indicators, std::vector<std::vector<int>>& adj_list);

int get_rdonly_map(
  const int num_blocks, const int threads_per_block, const std::vector<std::vector<int>>& adj_list,
  std::vector<std::vector<int>>& rdonly_map);

void reget_rdonly_map(
  const int num_blocks, const int max_rdonly_per_block,
  const std::vector<std::vector<int>>& rdonly_map, std::vector<int>& new_rdonly_map);

void get_nbors_bufs(
  const int block_threads, const std::vector<std::vector<Pt3>>& pts,
  const std::vector<std::vector<int>>& adj_list, const std::vector<std::vector<int>>& rd_only_idxs,
  std::vector<std::vector<int>>& nbors_bufs);

void write_to_file(
  const int rest_len, const int threads_per_block, const int num_blocks, const int max_pts,
  const int max_nbors_per_pt, const int max_nbors_per_block, const int max_nbors,
  const int max_rdonly_per_block, const int max_rdonly, const std::vector<Pt3>& pts,
  const std::vector<bool>& indicators, const std::vector<int>& rdonly_map,
  const std::vector<int>& nbors_map, const std::string file);

void print_sphere_stats(
  const std::vector<std::vector<Pt3>>& pts, const std::vector<std::vector<int>>& adj_list,
  const std::vector<std::vector<int>>& rd_only_idxs,
  const std::vector<std::vector<int>>& nbors_bufs);

#endif // CREATE_INPUT_HPP
