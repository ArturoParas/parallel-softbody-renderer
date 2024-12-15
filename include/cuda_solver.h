#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

struct GlobalConstants
{
  float particle_rad{1.f};
  float particle_mass{1.f};

  int spring_rest_len{4};
  float spring_k{2.f};
  float spring_damp{0.98};

  int intermediate_steps{8};
  float dt2_intermediate{0.1 * 0.1 / (8 * 8)};
  float g{-64.f};
  float width{1000.f};  // x
  float height{1000.f}; // y
  float depth{1000.f};  // z

  int particles_per_block{0};
  int num_blocks{0};
  int max_particles{0};
  int max_nbors_per_particle{0};
  int max_nbors_per_block{0};
  int max_nbors{0};
  int max_rdonly_per_block{0};
  int max_rdonly{0};

  float3* curr_particles{NULL};
  float3* prev_particles{NULL};
  uint16_t* rdonly_nbors{NULL};
  uint16_t* nbor_map{NULL};
};

void solver_update(GlobalConstants& h_params, float* h_curr_particles);

void solver_setup(
  GlobalConstants& h_params, const float* h_curr_particles, const uint16_t* h_rdonly_nbors,
  const uint16_t* h_nbor_map);

void solver_free(GlobalConstants& h_params);
