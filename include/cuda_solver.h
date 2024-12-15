#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

struct GlobalConstants
{
  float particle_rad{1.f};
  float particle_diameter{2.f};
  float particle_mass{1.f};

  int spring_rest_len{4};
  float spring_k{2.f};
  float spring_damp{0.98};

  int intermediate_steps{8};
  float dt{0.1};
  float dt_intermediate{0.1 / 8};
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
  int16_t* rdonly_nbors{NULL};
  int16_t* nbor_map{NULL};

  void set_dt(float dt_)
  {
    dt = dt_;
    update_dt_intermediate();
    update_dt2_intermediate();
  }

  void set_intermediate_steps(int intermediate_steps_)
  {
    intermediate_steps = intermediate_steps_;
    update_dt_intermediate();
    update_dt2_intermediate();
  }

  void set_dt_and_intermediate_steps(float dt_, int intermediate_steps_)
  {
    dt = dt_;
    intermediate_steps = intermediate_steps_;
    update_dt_intermediate();
    update_dt2_intermediate();
  }

  void set_particle_rad(float particle_rad_)
  {
    particle_rad = particle_rad_;
    update_diameter();
  }

private:
  void update_dt2_intermediate()
  {
    dt2_intermediate = dt * dt / (intermediate_steps * intermediate_steps);
  }

  void update_dt_intermediate()
  {
    dt_intermediate = dt / intermediate_steps;
  }

  void update_diameter()
  {
    particle_diameter = 2 * particle_rad;
  }
};

double solver_update(GlobalConstants& h_params, float* h_curr_particles);

void solver_setup(
  GlobalConstants& h_params, const float* h_curr_particles, const int16_t* h_rdonly_nbors,
  const int16_t* h_nbor_map);

void solver_free(GlobalConstants& h_params);
