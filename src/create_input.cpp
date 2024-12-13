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

std::vector<int> block_threads_to_dims(const int block_threads, const int rest_len)
{
  std::vector<int> dims;
  if (block_threads == 320) {
    dims.push_back(8 * rest_len);
    dims.push_back(8 * rest_len);
    dims.push_back(5 * rest_len);
  } else {
    std::cerr << "No conversion from given blocks per thread (" << block_threads
      << ") to block dims" << std::endl;
    exit(EXIT_FAILURE);
  }
  return dims;
}

void get_sphere_pts(
  const int block_threads, const int rad, const Pt3& center, const int rest_len,
  std::vector<std::vector<Pt3>>& pts, std::unordered_map<Pt3, int>& pt_idxs)
{
  int rad2 = rad * rad;
  std::vector<int> block_dims = block_threads_to_dims(block_threads, rest_len);
  int block_idx = 0;
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
                int idx = block_idx * block_threads + block_pts.size();
                pt_idxs.insert({Pt3(x + center.x, y + center.y, z + center.z), idx});
                block_pts.push_back(Pt3(x + center.x, y + center.y, z + center.z));
              }
            }
          }
        }
        if (block_pts.size() != 0) {
          pts.push_back(block_pts);
          block_idx++;
        }
      }
    }
  }
}

void get_adj_list(
  const int rest_len, const std::vector<std::vector<Pt3>>& pts,
  const std::unordered_map<Pt3, int>& pt_idxs, std::vector<std::vector<int>>& adj_list)
{
  for (const auto& block : pts) {
    for (const auto& pt : block) {
      std::vector<int> adj_pts;
      Pt3 pt_adj;
      for (int dz = -rest_len; dz <= rest_len; dz += rest_len) {
        pt_adj.z = pt.z + dz;
        for (int dy = -rest_len; dy <= rest_len; dy += rest_len) {
          pt_adj.y = pt.y + dy;
          for (int dx = -rest_len; dx <= rest_len; dx += rest_len) {
            pt_adj.x = pt.x + dx;
            if ((dz != 0 || dy != 0 || dx != 0) && (pt_idxs.count(pt_adj))) {
              adj_pts.push_back(pt_idxs.at(pt_adj));
            }
          }
        }
      }
      adj_list.push_back(adj_pts);
    }
  }
}

void get_rd_only_idxs(
  const int block_threads, const std::vector<std::vector<Pt3>>& pts,
  const std::vector<std::vector<int>>& adj_list, std::vector<std::vector<int>>& rd_only_idxs)
{
  int adj_start = 0;
  for (int block_idx = 0; block_idx < pts.size(); block_idx++) {
    int block_start = block_idx * block_threads;
    int block_end = block_start + pts.at(block_idx).size();
    std::set<int> block_rd_only_idxs;
    for (int off = 0; off < pts.at(block_idx).size(); off++) {
      for (const auto& adj_pt_idx : adj_list.at(adj_start + off)) {
        if (adj_pt_idx < block_start || block_end <= adj_pt_idx) {
          block_rd_only_idxs.insert(adj_pt_idx);
        }
      }
    }
    adj_start += pts.at(block_idx).size();
    rd_only_idxs.push_back(std::vector<int>(block_rd_only_idxs.begin(), block_rd_only_idxs.end()));
  }
}

void get_nbors_bufs(
  const int block_threads, const std::vector<std::vector<Pt3>>& pts,
  const std::vector<std::vector<int>>& adj_list, const std::vector<std::vector<int>>& rd_only_idxs,
  std::vector<std::vector<int>>& nbors_bufs)
{
  int adj_start = 0;
  for (int block_idx = 0; block_idx < pts.size(); block_idx++) {
    int block_start = block_idx * block_threads;
    int block_end = block_start + pts.at(block_idx).size();
    const std::vector<int>& block_rd_only_idxs = rd_only_idxs.at(block_idx);
    std::vector<int> nbors_buf;
    for (int nbor = 0; nbor < NBORS_PER_PT; nbor++) {
      for (int off = 0; off < pts.at(block_idx).size(); off++) {
        const std::vector<int>& nbor_pts = adj_list.at(adj_start + off);
        if (nbor < nbor_pts.size()) {
          int nbor_idx = nbor_pts.at(nbor);
          if (block_start <= nbor_idx && nbor_idx < block_end) {
            nbors_buf.push_back(nbor_idx - block_start);
          } else {
            auto it = std::lower_bound(
              block_rd_only_idxs.begin(), block_rd_only_idxs.end(), nbor_idx);
            nbors_buf.push_back(it - block_rd_only_idxs.begin() + block_threads);
          }
        } else {
          nbors_buf.push_back(-1);
        }
      }
    }
    adj_start += pts.at(block_idx).size();
    nbors_bufs.push_back(nbors_buf);
  }
}

void write_to_file(
  const int rad, const int rest_len, const std::vector<std::vector<Pt3>>& pts,
  const std::vector<std::vector<int>>& rd_only_idxs,
  const std::vector<std::vector<int>>& nbors_bufs, const std::string file)
{
  std::ofstream of("../inputs/" + file);

  // Sphere:
  of << rad << " " << rest_len << "\n";

  // Points!
  // [num blocks]
  of << pts.size() << "\n";
  for (const auto& block : pts) {
    // [num pts in block]
    of << block.size() << "\n";
    for (const auto& pt : block) {
      // [pt in block]
      of << pt.x << " " << pt.y << " " << pt.z << "\n";
    }
  }

  // Read only indices:
  for (const auto& block_rd_only_idxs : rd_only_idxs) {
    // [num read only pts in block]
    of << block_rd_only_idxs.size() << "\n";
    for (const auto& rd_only_idx : block_rd_only_idxs) {
      // [idx of read only point in global pts buff]
      of << rd_only_idx << " ";
    }
    // next block
    of << "\n";
  }

  // Nbors bufs:
  for (const auto& nbors_buf : nbors_bufs) {
    for (const auto& nbor : nbors_buf) {
      // [idx of a points neighbor in nbors buff]
      of << nbor << " ";
    }
    // next block
    of << "\n";
  }
}

void print_sphere_stats(
  const std::vector<std::vector<Pt3>>& pts, const std::vector<std::vector<int>>& adj_list,
  const std::vector<std::vector<int>>& rd_only_idxs,
  const std::vector<std::vector<int>>& nbors_bufs)
{
  int num_blocks = pts.size();
  int num_pts = 0;
  int max_pts = 0;
  int min_pts = __INT_MAX__;
  for (const auto& block : pts) {
    num_pts += block.size();
    max_pts = std::max(max_pts, (int)block.size());
    min_pts = std::min(min_pts, (int)block.size());
  }
  std::cout << "num blocks = " << num_blocks << std::endl;
  std::cout << "num points = " << num_pts << std::endl;
  std::cout << "num points per block = " << (float)num_pts / num_blocks << std::endl;
  std::cout << "max points in a block = " << max_pts << std::endl;
  std::cout << "min points in a block = " << min_pts << std::endl;

  int num_adj_entries = adj_list.size();
  float num_springs = 0;
  int max_nbors = 0;
  int min_nbors = NBORS_PER_PT;
  for (const auto& adj_pts : adj_list) {
    num_springs += adj_pts.size();
    max_nbors = std::max(max_nbors, (int)adj_pts.size());
    min_nbors = std::min(min_nbors, (int)adj_pts.size());
  }
  num_springs /= 2;
  std::cout << std::endl;
  std::cout << "num adjacency entries = " << num_adj_entries << std::endl;
  std::cout << "num springs = " << num_springs << std::endl;
  std::cout << "num springs per block = " << num_springs / num_blocks << std::endl;
  std::cout << "num springs per point = " << num_springs / num_pts << std::endl;
  std::cout << "max neighbors for a point = " << max_nbors << std::endl;
  std::cout << "min neighbors for a point = " << min_nbors << std::endl;

  int num_rd_only_entries = rd_only_idxs.size();
  int num_rd_only = 0;
  int max_rd_only = 0;
  int min_rd_only = __INT_MAX__;
  for (const auto& block_rd_only_idxs : rd_only_idxs) {
    num_rd_only += block_rd_only_idxs.size();
    max_rd_only = std::max(max_rd_only, (int)block_rd_only_idxs.size());
    min_rd_only = std::min(min_rd_only, (int)block_rd_only_idxs.size());
  }
  std::cout << std::endl;
  std::cout << "num read only entries = " << num_rd_only_entries << std::endl;
  std::cout << "num read only points = " << num_rd_only << std::endl;
  std::cout << "num read only points per block = " << (float)num_rd_only / num_blocks << std::endl;
  std::cout << "max read only points in a block = " << max_rd_only << std::endl;
  std::cout << "min read only points in a block = " << min_rd_only << std::endl;

  int num_nbors_bufs_entries = nbors_bufs.size();
  num_springs = 0;
  int num_garbage = 0;
  float max_springs = 0;
  int max_garbage = 0;
  int max_buf_size = 0;
  int max_nbor = 0;
  for (const auto& nbors_buf : nbors_bufs) {
    int block_springs = 0;
    int block_garbage = 0;
    for (const auto& nbor : nbors_buf) {
      if (nbor == -1) {
        num_garbage++;
        block_garbage++;
      } else {
        num_springs++;
        block_springs++;
      }
      max_nbor = std::max(max_nbor, nbor);
    }
    max_springs = std::max(max_springs, (float)block_springs);
    max_garbage = std::max(max_garbage, block_garbage);
    max_buf_size = std::max(max_buf_size, (int)nbors_buf.size());
  }
  num_springs /= 2;
  max_springs /= 2;
  std::cout << std::endl;
  std::cout << "num neighbors bufs entires = " << num_nbors_bufs_entries << std::endl;
  std::cout << "num springs = " << num_springs << std::endl;
  std::cout << "num garbage = " << num_garbage << std::endl;
  std::cout << "num garbage per block = " << (float)num_garbage / num_blocks << std::endl;
  std::cout << "max springs in a block = " << max_springs << std::endl;
  std::cout << "max garbage in a block = " << max_garbage << std::endl;
  std::cout << "max buffer size = " << max_buf_size << std::endl;
  std::cout << "max neighbor index = " << max_nbor << std::endl;
}

int main(int argc, char* argv[])
{
  std::string file = "sphere.txt";
  int rad = 53;
  int rest_len = 4;
  Pt3 center(0, 0, 0);
  int block_threads = 320;

  int opt;
  char* end;
  while ((opt = getopt(argc, argv, "r:l:")) != EOF) {
    switch (opt) {
    case 'r':
      rad = strtol(optarg, &end, 10);
      break;
    case 'l':
      rest_len = strtol(optarg, &end, 10);
      break;
    default:
      std::cout << "Given optargs not valid" << std::endl;
      break;
    }
  }

  std::vector<std::vector<Pt3>> pts;
  std::unordered_map<Pt3, int> pt_idxs;
  std::vector<std::vector<int>> adj_list;
  std::vector<std::vector<int>> rd_only_idxs;
  std::vector<std::vector<int>> nbors_bufs;

  get_sphere_pts(block_threads, rad, center, rest_len, pts, pt_idxs);
  get_adj_list(rest_len, pts, pt_idxs, adj_list);
  get_rd_only_idxs(block_threads, pts, adj_list, rd_only_idxs);
  get_nbors_bufs(block_threads, pts, adj_list, rd_only_idxs, nbors_bufs);
  write_to_file(rad, rest_len, pts, rd_only_idxs, nbors_bufs, file);
  print_sphere_stats(pts, adj_list, rd_only_idxs, nbors_bufs);
  return 0;
}
