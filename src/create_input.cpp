#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>
#include <iterator>

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
          pts.emplace_back(Pt3(tx, ty, tz));
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
              springs.emplace_back(Spring(idx, pt_to_idx.at(other)));
            }
          }
        }
      }
    }
    it = pt_to_idx.erase(it);
  }
}

void print_sphere_stats(const std::vector<Pt3>& pts, const std::vector<Spring>& springs)
{
  std::cout << "num circles = " << pts.size() << std::endl;
  std::cout << "num springs = " << springs.size() << std::endl;
  std::cout << "num springs / circle = " << (float)springs.size() / (float)pts.size() << std::endl;
}

void write_to_file(
  const std::vector<Pt3>& pts, const std::vector<Spring>& springs, const std::string file)
{
  std::ofstream of("../inputs/" + file);
  of << pts.size() << "\n";
  for (const auto& pt : pts) {
    of << pt.x << " " << pt.y << " " << pt.z << "\n";
  }
  of << springs.size() << "\n";
  for (const auto& spring : springs) {
    of << spring.idx1 << " " << spring.idx2 << "\n";
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
  get_springs(rest_len, pt_to_idx, springs);
  print_sphere_stats(pts, springs);
  write_to_file(pts, springs, file);
  return 0;
}
