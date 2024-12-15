#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <algorithm>
#include <cmath>
#include <getopt.h>

#include "../include/create_input.hpp"

#define NBORS_PER_PT 26

Pt3::Pt3(const float x_, const float y_, const float z_) : x(x_), y(y_), z(z_) {};

Pt3::Pt3() : x(0), y(0), z(0) {};

bool Pt3::operator==(const Pt3& other) const
{
  return (x == other.x && y == other.y && z == other.z);
}

std::size_t std::hash<Pt3>::operator()(const Pt3& pt) const
{
  std::size_t hx = std::hash<float>{}(pt.x);
  std::size_t hy = std::hash<float>{}(pt.y);
  std::size_t hz = std::hash<float>{}(pt.z);
  return (hx ^ (hy << 1) >> 1) ^ (hz << 1);
}

std::vector<int> threads_per_block_to_dims(const int threads_per_block, const int rest_len)
{
  std::vector<int> dims;
  if (threads_per_block == 672) {
    dims.push_back(12 * rest_len);
    dims.push_back(8 * rest_len);
    dims.push_back(7 * rest_len);
  } else if (threads_per_block == 640) {
    dims.push_back(10 * rest_len);
    dims.push_back(8 * rest_len);
    dims.push_back(8 * rest_len);
  } else if (threads_per_block == 320) {
    dims.push_back(8 * rest_len);
    dims.push_back(8 * rest_len);
    dims.push_back(5 * rest_len);
  } else if (threads_per_block == 32) {
    dims.push_back(4 * rest_len);
    dims.push_back(4 * rest_len);
    dims.push_back(2 * rest_len);
  } else if (threads_per_block == 16) {
    dims.push_back(4 * rest_len);
    dims.push_back(2 * rest_len);
    dims.push_back(2 * rest_len);
  } else if (threads_per_block == 8) {
    dims.push_back(2 * rest_len);
    dims.push_back(2 * rest_len);
    dims.push_back(2 * rest_len);
  } else if (threads_per_block == 4) {
    dims.push_back(2 * rest_len);
    dims.push_back(2 * rest_len);
    dims.push_back(1 * rest_len);
  } else if (threads_per_block == 1) {
    dims.push_back(1 * rest_len);
    dims.push_back(1 * rest_len);
    dims.push_back(1 * rest_len);
  } else {
    std::cerr << "No conversion from given blocks per thread (" << threads_per_block
      << ") to block dims" << std::endl;
    exit(EXIT_FAILURE);
  }
  return dims;
}

int get_cube_pts(
  const int threads_per_block, const int rad, const Pt3& center, const int rest_len,
  std::vector<std::vector<Pt3>>& pts)
{
  std::vector<int> block_dims = threads_per_block_to_dims(threads_per_block, rest_len);
  for (int z_block_off = -rad; z_block_off <= rad; z_block_off += block_dims.at(2)) {
    int z_block_next = std::min(z_block_off + block_dims.at(2), rad);
    for (int y_block_off = -rad; y_block_off <= rad; y_block_off += block_dims.at(1)) {
      int y_block_next = std::min(y_block_off + block_dims.at(1), rad);
      for (int x_block_off = -rad; x_block_off <= rad; x_block_off += block_dims.at(0)) {
        std::vector<Pt3> block_pts;
        int x_block_next = std::min(x_block_off + block_dims.at(0), rad);
        for (int z = z_block_off; z < z_block_next; z += rest_len) {
          int z2 = z * z;
          for (int y = y_block_off; y < y_block_next; y += rest_len) {
            int y2 = y * y;
            for (int x = x_block_off; x < x_block_next; x += rest_len) {
              int x2 = x * x;
              block_pts.push_back(Pt3(x + center.x, y + center.y, z + center.z));
            }
          }
        }
        if (block_pts.size() != 0) {
          pts.push_back(block_pts);
        }
      }
    }
  }
  return pts.size();
}

int get_good_cube_pts(
  const int threads_per_block, const int base, const Pt3& center, const int rest_len,
  std::vector<std::vector<Pt3>>& pts)
{
  float sqrt3 = sqrt(3);
  float sqrt23 = sqrt(2 / 3);
  std::vector<Pt3> block;
  for (int i = 0; i < base; i += rest_len) {
    ;
  }
  return pts.size();
}

int get_pyramid_pts(
  const int threads_per_block, const int base, const Pt3& center, const int rest_len,
  std::vector<std::vector<Pt3>>& pts)
{
  ;
}

int get_sphere_pts(
  const int threads_per_block, const int rad, const Pt3& center, const int rest_len,
  std::vector<std::vector<Pt3>>& pts)
{
  int rad2 = rad * rad;
  std::vector<int> block_dims = threads_per_block_to_dims(threads_per_block, rest_len);
  for (int z_block_off = -rad; z_block_off <= rad; z_block_off += block_dims.at(2)) {
    int z_block_next = std::min(z_block_off + block_dims.at(2), rad);
    for (int y_block_off = -rad; y_block_off <= rad; y_block_off += block_dims.at(1)) {
      int y_block_next = std::min(y_block_off + block_dims.at(1), rad);
      for (int x_block_off = -rad; x_block_off <= rad; x_block_off += block_dims.at(0)) {
        std::vector<Pt3> block_pts;
        int x_block_next = std::min(x_block_off + block_dims.at(0), rad);
        for (int z = z_block_off; z < z_block_next; z += rest_len) {
          int z2 = z * z;
          for (int y = y_block_off; y < y_block_next; y += rest_len) {
            int y2 = y * y;
            for (int x = x_block_off; x < x_block_next; x += rest_len) {
              int x2 = x * x;
              if (x2 + y2 + z2 <= rad2) {
                block_pts.push_back(Pt3(x + center.x, y + center.y, z + center.z));
              }
            }
          }
        }
        if (block_pts.size() != 0) {
          pts.push_back(block_pts);
        }
      }
    }
  }
  return pts.size();
}

void reget_pts(
  const int num_blocks, const int threads_per_block, const std::vector<std::vector<Pt3>>& pts,
  std::vector<Pt3>& new_pts, std::unordered_map<Pt3, int>& map, std::vector<bool>& indicators)
{
  for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    if (block_idx < pts.size()) {
      const std::vector<Pt3>& block_pts = pts.at(block_idx);
      for (int pt = 0; pt < threads_per_block; pt++) {
        if (pt < block_pts.size()) {
          map.insert({block_pts.at(pt), new_pts.size()});
          new_pts.push_back(block_pts.at(pt));
          indicators.push_back(true);
        } else {
          new_pts.push_back(Pt3());
          indicators.push_back(false);
        }
      }
    } else {
      std::cerr << "more blocks than found in get_pts, wrong!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

int get_adj_list(
  const int rest_len, const std::vector<Pt3>& pts, const std::unordered_map<Pt3, int>& idx_map,
  const std::vector<bool>& indicators, std::vector<std::vector<int>>& adj_list)
{
  int max_nbors_per_pt = 0;
  for (int pt_idx = 0; pt_idx < pts.size(); pt_idx++) {
    if (indicators.at(pt_idx)) {
      const Pt3& pt = pts.at(pt_idx);
      std::vector<int> nbors;
      Pt3 nbor;
      for (int dz = -rest_len; dz <= rest_len; dz += rest_len) {
        nbor.z = pt.z + dz;
        for (int dy = -rest_len; dy <= rest_len; dy += rest_len) {
          nbor.y = pt.y + dy;
          for (int dx = -rest_len; dx <= rest_len; dx += rest_len) {
            nbor.x = pt.x + dx;
            if ((dz != 0 || dy != 0 || dx != 0) && (idx_map.count(nbor))) {
              nbors.push_back(idx_map.at(nbor));
            }
          }
        }
      }
      max_nbors_per_pt = std::max(max_nbors_per_pt, (int)nbors.size());
      adj_list.push_back(nbors);
    } else {
      adj_list.push_back(std::vector<int>());
    }
  }
  return max_nbors_per_pt;
}

int get_rdonly_map(
  const int num_blocks, const int threads_per_block, const std::vector<std::vector<int>>& adj_list,
  std::vector<std::vector<int>>& rdonly_map)
{
  int max_rdonly_per_block = 0;
  for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int block_off = block_idx * threads_per_block;
    std::set<int> map;
    for (int pt_off = 0; pt_off < threads_per_block; pt_off++) {
      int pt_idx = block_off + pt_off;
      for (const auto& nbor : adj_list.at(pt_idx)) {
        if (nbor < block_off || block_off + threads_per_block <= nbor) {
          map.insert(nbor);
        }
      }
    }
    max_rdonly_per_block = std::max(max_rdonly_per_block, (int)map.size());
    rdonly_map.push_back(std::vector<int>(map.begin(), map.end()));
  }
  return max_rdonly_per_block;
}

void reget_rdonly_map(
  const int num_blocks, const int max_rdonly_per_block,
  const std::vector<std::vector<int>>& rdonly_map, std::vector<int>& new_rdonly_map)
{
  for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    if (block_idx < rdonly_map.size()) {
      const std::vector<int>& map = rdonly_map.at(block_idx);
      for (int rdonly_idx = 0; rdonly_idx < max_rdonly_per_block; rdonly_idx++) {
        if (rdonly_idx < map.size()) {
          new_rdonly_map.push_back(map.at(rdonly_idx));
        } else {
          new_rdonly_map.push_back(-1);
        }
      }
    } else {
      std::cerr << "more blocks found than in rdonly_map, wrong!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

void get_nbors_map(
  const int num_blocks, const int threads_per_block, const int max_nbors_per_pt,
  const std::vector<bool>& indicators, const std::vector<std::vector<int>>& adj_list,
  const std::vector<std::vector<int>>& rdonly_map, std::vector<std::vector<int>>& nbors_map)
{
  for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int block_off = block_idx * threads_per_block;
    const std::vector<int>& block_rdonly_map = rdonly_map.at(block_idx);
    std::vector<int> map;
    for (int nbor = 0; nbor < max_nbors_per_pt; nbor++) {
      for (int pt_off = 0; pt_off < threads_per_block; pt_off++) {
        int pt_idx = block_off + pt_off;
        if (pt_idx < indicators.size()) {
          if (indicators.at(pt_idx)) {
            const std::vector<int>& nbors = adj_list.at(pt_idx);
            if (nbor < nbors.size()) {
              int nbor_idx = nbors.at(nbor);
              if (block_off <= nbor_idx && nbor_idx < block_off + threads_per_block) {
                map.push_back(nbor_idx - block_off);
              } else {
                auto it = std::lower_bound(
                  block_rdonly_map.begin(), block_rdonly_map.end(), nbor_idx);
                map.push_back(it - block_rdonly_map.begin() + threads_per_block);
              }
            } else {
              map.push_back(-1);
            }
          } else {
            map.push_back(-1);
          }
        } else {
          map.push_back(-1);
        }
      }
    }
    nbors_map.push_back(map);
  }
}

void reget_nbors_map(
  const std::vector<std::vector<int>>& nbors_map, std::vector<int>& new_nbors_map)
{
  for (const auto& map : nbors_map) {
    for (const auto& idx : map) {
      new_nbors_map.push_back(idx);
    }
  }
}

void write_to_file(
  const int rest_len, const int threads_per_block, const int num_blocks, const int max_pts,
  const int max_nbors_per_pt, const int max_nbors_per_block, const int max_nbors,
  const int max_rdonly_per_block, const int max_rdonly, const int rad, const std::vector<Pt3>& pts,
  const std::vector<bool>& indicators, const std::vector<int>& rdonly_map,
  const std::vector<int>& nbors_map, const std::string file)
{
  std::cout << "FOR COMPILE TIME DEFINES:" << std::endl;
  std::cout << "Particles per block = " << threads_per_block << std::endl;
  std::cout << "Max rdonly per block = " << max_rdonly_per_block << std::endl;
  std::cout << "Max nbors per block = " << max_nbors_per_block << std::endl;
  std::cout << std::endl;
  std::cout << "FOR BENCHMARKING:" << std::endl;
  std::cout << "Num blocks = " << num_blocks << std::endl;
  std::cout << "Max num particles = " << max_pts << std::endl;

  std::ofstream of ("../inputs/" + file);

  of << rest_len << " " << threads_per_block << " " << num_blocks << " " << max_pts << " "
    << max_nbors_per_pt << " " << max_nbors_per_block << " " << max_nbors << " "
    << max_rdonly_per_block << " " << max_rdonly << " " << rad << "\n";

  for (const auto& pt : pts) {
    of << pt.x << " " << pt.y << " " << pt.z << "\n";
  }
  for (const auto& indicator : indicators) {
    of << indicator << " ";
  }
  of << "\n";

  for (const auto& g_idx : rdonly_map) {
    of << g_idx << " ";
  }
  of << "\n";

  for (const auto& s_idx : nbors_map) {
    of << s_idx << " ";
  }
  of << "\n";
}

int main(int argc, char* argv[])
{
  std::string file = "cube.txt";
  int shape = 0;
  int rad = 53;
  int rest_len = 4;
  Pt3 center(0, 0, 0);
  int threads_per_block = 320;

  int opt;
  char* end;
  while ((opt = getopt(argc, argv, "r:l:t:s:")) != EOF) {
    switch (opt) {
    case 'r':
      rad = strtol(optarg, &end, 10);
      break;
    case 'l':
      rest_len = strtol(optarg, &end, 10);
      break;
    case 't':
      threads_per_block = strtol(optarg, &end, 10);
      break;
    case 's':
      if (*optarg == 'c') {
        file = "cube.txt";
        shape = 0;
      } else if (*optarg == 'p') {
        file = "pyramid.txt";
        shape = 1;
      } else if (*optarg == 's') {
        file = "sphere.txt";
        shape = 2;
      }
      break;
    default:
      std::cout << "Given optargs not valid" << std::endl;
      break;
    }
  }

  std::vector<std::vector<Pt3>> pts;
  std::vector<Pt3> new_pts;
  std::unordered_map<Pt3, int> idx_map;
  std::vector<bool> indicators;
  std::vector<std::vector<int>> adj_list;
  std::vector<std::vector<int>> rdonly_map;
  std::vector<int> new_rdonly_map;
  std::vector<std::vector<int>> nbors_map;
  std::vector<int> new_nbors_map;

  int num_blocks = 0;
  if (shape == 0) {
    num_blocks = get_cube_pts(threads_per_block, rad, center, rest_len, pts);
  }  else if (shape == 1) {
     int num_blocks = get_pyramid_pts(threads_per_block, rad, center, rest_len, pts);
  } else if (shape == 2) {
    num_blocks = get_sphere_pts(threads_per_block, rad, center, rest_len, pts);
  }

  int max_pts = threads_per_block * num_blocks;
  reget_pts(num_blocks, threads_per_block, pts, new_pts, idx_map, indicators);

  int max_nbors_per_pt = get_adj_list(rest_len, new_pts, idx_map, indicators, adj_list);
  int max_nbors_per_block = threads_per_block * max_nbors_per_pt;
  int max_nbors = max_nbors_per_block * num_blocks;

  int max_rdonly_per_block = get_rdonly_map(num_blocks, threads_per_block, adj_list, rdonly_map);
  int max_rdonly = max_rdonly_per_block * num_blocks;
  reget_rdonly_map(num_blocks, max_rdonly_per_block, rdonly_map, new_rdonly_map);

  get_nbors_map(num_blocks, threads_per_block, max_nbors_per_pt, indicators, adj_list, rdonly_map,
    nbors_map);
  reget_nbors_map(nbors_map, new_nbors_map);

  write_to_file(rest_len, threads_per_block, num_blocks, max_pts, max_nbors_per_pt,
    max_nbors_per_block, max_nbors, max_rdonly_per_block, max_rdonly, rad, new_pts, indicators,
    new_rdonly_map, new_nbors_map, file);
  return 0;
}
