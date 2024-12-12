#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <algorithm>

#include "../include/create_input.hpp"

Pt3::Pt3(const float x_, const float y_, const float z_) : x(x_), y(y_), z(z_) {};

bool Pt3::operator==(const Pt3& other) const
{
  return (x == other.x && y == other.y && z == other.z);
}

Spring::Spring(const std::size_t idx1_, const std::size_t idx2_) : idx1(idx1_), idx2(idx2_) {};

std::size_t std::hash<Pt3>::operator()(const Pt3& pt) const
{
  std::size_t hx = std::hash<float>{}(pt.x);
  std::size_t hy = std::hash<float>{}(pt.y);
  std::size_t hz = std::hash<float>{}(pt.z);
  return (hx ^ (hy << 1) >> 1) ^ (hz << 1);
}

/** TODO: Traverse by blocks with volume of num update circles per block */
void get_sphere_pts(
  const int rad, const Pt3& center, const int rest_len, std::vector<Pt3>& pts,
  std::unordered_map<Pt3, std::size_t>& pt_to_idx)
{
  int rad2 = rad * rad;
  for (int tx = -rad; tx <= rad; tx += rest_len) {
    int x = tx - center.x;
    int x2 = x * x;
    for (int ty = -rad; ty <= rad; ty += rest_len) {
      int y = ty - center.y;
      int y2 = y * y;
      for (int tz = -rad; tz <= rad; tz += rest_len) {
        int z = tz - center.z;
        int z2 = z * z;
        if (x2 + y2 + z2 <= rad2) {
          pt_to_idx.insert({Pt3(tx, ty, tz), pt_to_idx.size()});
          pts.push_back(Pt3(tx, ty, tz));
        }
      }
    }
  }
}

void get_springs(
  const int rest_len, std::unordered_map<Pt3, std::size_t>& pt_to_idx,
  std::vector<Spring>& springs)
{
  auto it = pt_to_idx.begin();
  while (it != pt_to_idx.end()) {
    Pt3 pt = it->first;
    std::size_t idx = it->second;
    Pt3 other(pt.x, pt.y, pt.z);
    for (int dz = -rest_len; dz <= rest_len; dz += rest_len) {
      other.z = pt.z + dz;
      for (int dy = -rest_len; dy <= rest_len; dy += rest_len) {
        other.y = pt.y + dy;
        for (int dx = -rest_len; dx <= rest_len; dx += rest_len) {
          if (dz != 0 || dy != 0 || dx != 0) {
            other.x = pt.x + dx;
            if (pt_to_idx.count(other)) {
              springs.push_back(Spring(idx, pt_to_idx.at(other)));
            }
          }
        }
      }
    }
    it = pt_to_idx.erase(it);
  }
}

void get_adjacency_list(
  const int rest_len, const std::unordered_map<Pt3, std::size_t>& pt_to_idx,
  std::vector<std::vector<std::size_t>>& adjacency_list)
{
  for (const auto& i : pt_to_idx) {
    Pt3 pt = i.first;
    std::size_t idx = i.second;
    Pt3 other(pt.x, pt.y, pt.z);
    std::vector<std::size_t>& adjacent_indices = adjacency_list[idx];
    for (int dz = -rest_len; dz <= rest_len; dz += rest_len) {
      other.z = pt.z + dz;
      for (int dy = -rest_len; dy <= rest_len; dy += rest_len) {
        other.y = pt.y + dy;
        for (int dx = -rest_len; dx <= rest_len; dx += rest_len) {
          if (dz != 0 || dy != 0 || dx != 0) {
            other.x = pt.x + dx;
            if (pt_to_idx.count(other)) {
              adjacent_indices.push_back(pt_to_idx.at(other));
            }
          }
        }
      }
    }
  }
}

std::size_t get_particle_idx_bufs(
  const int update_particles_per_block, const std::size_t num_pts,
  const std::vector<std::vector<std::size_t>>& adjacency_list,
  std::vector<std::set<std::size_t>>& rd_only_particle_idx_bufs)
{
  std::size_t max_rd_only_particles = 0;
  for (std::size_t block_offset = 0; block_offset < num_pts; block_offset += update_particles_per_block) {
    // std::vector<Pt3>& particle_idx_buf = particle_idx_bufs[block_offset / update_particle_per_block];
    std::set<std::size_t> rd_only_particle_idx_set;
    std::size_t loop_guard = std::min(block_offset + update_particles_per_block, num_pts); 
    for (std::size_t i = block_offset; i < loop_guard; i++) {
      for (const auto& pt_idx : adjacency_list[i]) {
        if (pt_idx < block_offset || block_offset + update_particles_per_block <= pt_idx) {
          rd_only_particle_idx_set.insert(pt_idx);
        }
      }
    }
    max_rd_only_particles = std::max(max_rd_only_particles, rd_only_particle_idx_set.size());
    rd_only_particle_idx_bufs.push_back(rd_only_particle_idx_set);
  }
  return max_rd_only_particles;
}

void get_nbors_buf(
  const int update_particles_per_block,
  const std::vector<std::vector<std::size_t>>& adjacency_list,
  std::vector<std::set<std::size_t>>& rd_only_particle_idx_bufs)
{
  ;
}

void print_sphere_stats(
  const std::vector<Pt3>& pts, const std::vector<Spring>& springs,
  const std::size_t max_rd_only_particles)
{
  std::cout << "num pts = " << pts.size() << std::endl;
  std::cout << "num springs = " << springs.size() << std::endl;
  std::cout << "num springs / pts = " << (float)springs.size() / (float)pts.size() << std::endl;
  std::cout << "max number of read only particles = " << max_rd_only_particles << std::endl;
}

void write_to_file(
  const std::vector<Pt3>& pts, const std::vector<std::vector<std::size_t>>& adjacency_list,
  const std::string file)
{
  std::ofstream of("../inputs/" + file);
  of << pts.size() << "\n";
  for (const auto& pt : pts) {
    of << pt.x << " " << pt.y << " " << pt.z << "\n";
  }
  for (const auto& adjacent_indices : adjacency_list) {
    for (const auto& idx : adjacent_indices) {
      of << idx << " ";
    }
    of << "\n";
  }
}

int main(int argc, char* argv[])
{
  if (argc > 7) {
    std::cerr << "Too many arguments" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string file = "sphere.txt";
  int rad = 53;
  int rest_len = 4;
  Pt3 center(0, 0, 0);

  if (argc > 1) {
    file = argv[1];
  }
  if (argc > 2) {
    char* end;
    rad = strtol(argv[2], &end, 10);
  }
  if (argc > 3) {
    char* end;
    rest_len = strtol(argv[3], &end, 10);
  }
  if (argc == 6) {
    char* end;
    center.x = strtol(argv[4], &end, 10);
    center.y = strtol(argv[5], &end, 10);
    center.z = strtol(argv[6], &end, 10);
  }

  std::vector<Pt3> pts;
  std::unordered_map<Pt3, std::size_t> pt_to_idx;
  std::vector<Spring> springs;
  get_sphere_pts(rad, center, rest_len, pts, pt_to_idx);
  std::vector<std::vector<std::size_t>> adjacency_list(pts.size());
  get_adjacency_list(rest_len, pt_to_idx, adjacency_list);
  /** TODO: Analyze what update_particles_per_block should be */
  int update_particles_per_block = 128;
  std::vector<std::set<std::size_t>> rd_only_particle_idx_bufs;
  std::size_t max_rd_only_particles = get_particle_idx_bufs(
    update_particles_per_block, pts.size(), adjacency_list, rd_only_particle_idx_bufs);
  get_springs(rest_len, pt_to_idx, springs);
  print_sphere_stats(pts, springs, max_rd_only_particles);
  write_to_file(pts, adjacency_list, file);
  return 0;
}
