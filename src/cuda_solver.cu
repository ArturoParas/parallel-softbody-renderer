#include <stdio.h>
#include <iostream>

#include "../include/cuda_solver.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)
#define abs(a) (a < 0 ?  -a : a)

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

#define PARTICLES_PER_BLOCK 320
#define MAX_RDONLY_PER_BLOCK 380
#define MAX_NBORS_PER_BLOCK 8320

__constant__ GlobalConstants d_params;

__device__ __inline__ float3 update_iteration(
  const int t_idx, const int b_idx, const int d_off, const int d_idx, float3 prev_particle,
  float3* s_curr_particles, const int16_t* s_nbor_map)
{
  float3 curr_particle = s_curr_particles[t_idx];
  float3 next_particle = curr_particle;

  float dx = curr_particle.x - prev_particle.x;
  float dy = curr_particle.y - prev_particle.y;
  float dz = curr_particle.z - prev_particle.z;
  float vel_dist = sqrtf(dx * dx + dy * dy + dz * dz);

  next_particle.x += dx;
  next_particle.y += dy;
  next_particle.z += dz;

  for (int nbor = 0; nbor < d_params.max_nbors_per_particle; nbor++) {
    int n_key = s_nbor_map[nbor * d_params.particles_per_block + t_idx];
    if (n_key >= 0) {
      float3 nbor_particle = s_curr_particles[n_key];

      float disp_x = nbor_particle.x - curr_particle.x;
      float disp_y = nbor_particle.y - curr_particle.y;
      float disp_z = nbor_particle.z - curr_particle.z;
      float dist = sqrtf(disp_x * disp_x + disp_y * disp_y + disp_z * disp_z);
      float damp_dir = dx * disp_x + dy * disp_y + dz * disp_z;
      if (damp_dir != 0) {
        damp_dir /= abs(damp_dir);
      }

      float spring_force = d_params.spring_k * (dist - d_params.spring_rest_len);
      float damp_impulse = -d_params.spring_damp * vel_dist * damp_dir;
      float acc_dist = (spring_force * d_params.dt2_intermediate + damp_impulse * d_params.dt_intermediate) / (d_params.particle_mass * dist);

      next_particle.x += acc_dist * disp_x;
      next_particle.y += acc_dist * disp_y;
      next_particle.z += acc_dist * disp_z + d_params.g * d_params.dt2_intermediate;
    }
  }
  __syncthreads();

  s_curr_particles[t_idx] = next_particle;
  __syncthreads();

  prev_particle = curr_particle;
  curr_particle = next_particle;

  for (int nbor = 0; nbor < d_params.max_nbors_per_particle; nbor++) {
    int n_key = s_nbor_map[nbor * d_params.particles_per_block + t_idx];
    if (n_key >= 0) {
      float3 nbor_particle = s_curr_particles[n_key];

      float disp_x = nbor_particle.x - curr_particle.x;
      float disp_y = nbor_particle.y - curr_particle.y;
      float disp_z = nbor_particle.z - curr_particle.z;
      float dist = sqrtf(disp_x * disp_x + disp_y * disp_y + disp_z * disp_z);
      float overlap = dist - d_params.particle_diameter;
      if (overlap < 0) {
        float move_amt = overlap * 0.5 / dist;
        next_particle.x += disp_x * move_amt;
        next_particle.y += disp_y * move_amt;
        next_particle.z += disp_z * move_amt;
      }
    }
  }
  __syncthreads();

  next_particle.x = min(max(next_particle.x, -d_params.width / 2 + d_params.particle_rad),
    d_params.width / 2 - d_params.particle_rad);
  next_particle.y = min(max(next_particle.y, -d_params.height / 2 + d_params.particle_rad),
    d_params.height / 2 - d_params.particle_rad);
  next_particle.z = min(max(next_particle.z, -d_params.depth / 2 + d_params.particle_rad),
    d_params.depth / 2 - d_params.particle_rad);
  s_curr_particles[t_idx] = next_particle;

  return prev_particle;
}

__global__ void update_kernel()
{
  int t_idx = threadIdx.x;
  int b_idx = blockIdx.x;
  int d_off = b_idx * blockDim.x;
  int d_idx = d_off + t_idx;

  __shared__ float3 s_curr_particles[PARTICLES_PER_BLOCK + MAX_RDONLY_PER_BLOCK];
  __shared__ int16_t s_nbor_map[MAX_NBORS_PER_BLOCK];

  float3 prev_particle = d_params.prev_particles[d_idx];
  s_curr_particles[t_idx] = d_params.curr_particles[d_idx];

  for (int nbor = 0; nbor < d_params.max_nbors_per_particle; nbor++) {
    int s_idx = nbor * d_params.particles_per_block + t_idx;
    int n_key = d_params.nbor_map[d_off * d_params.max_nbors_per_particle + s_idx];
    s_nbor_map[s_idx] = n_key;
    if (n_key >= d_params.particles_per_block) {
      s_curr_particles[n_key] = d_params.curr_particles[d_params.rdonly_nbors[b_idx * d_params.max_rdonly_per_block + n_key - d_params.particles_per_block]];
    }
  }
  __syncthreads();

  for (int i = 0; i < d_params.intermediate_steps; i++) {
    prev_particle = update_iteration(t_idx, b_idx, d_off, d_idx, prev_particle, s_curr_particles, s_nbor_map);
    __syncthreads();
  }

  d_params.prev_particles[d_idx] = prev_particle;
  d_params.curr_particles[d_idx] = s_curr_particles[t_idx];
}

void solver_update(GlobalConstants& h_params, float* h_curr_particles)
{
  update_kernel<<<h_params.num_blocks, h_params.particles_per_block>>>();
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(h_curr_particles, h_params.curr_particles,
    h_params.max_particles * sizeof(float3), cudaMemcpyDeviceToHost));

  // printf("\n");
  // for (int i = 0; i < h_params.max_particles * 3; i += 3) {
  //   printf("h_curr_particles[%d] = (%f, %f, %f)\n", i / 3, h_curr_particles[i], h_curr_particles[i + 1], h_curr_particles[i + 2]);
  // }
}

void solver_setup(
  GlobalConstants& h_params, const float* h_curr_particles, const int16_t* h_rdonly_nbors,
  const int16_t* h_nbor_map)
{
  cudaCheckError(cudaMalloc(&h_params.curr_particles, h_params.max_particles * sizeof(float3)));
  cudaCheckError(cudaMalloc(&h_params.prev_particles, h_params.max_particles * sizeof(float3)));
  cudaCheckError(cudaMalloc(&h_params.rdonly_nbors, h_params.max_rdonly * sizeof(int16_t)));
  cudaCheckError(cudaMalloc(&h_params.nbor_map, h_params.max_nbors * sizeof(int16_t)));
  
  cudaCheckError(cudaMemcpy(h_params.curr_particles, h_curr_particles,
    h_params.max_particles * sizeof(float3), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(h_params.prev_particles, h_curr_particles,
    h_params.max_particles * sizeof(float3), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(h_params.rdonly_nbors, h_rdonly_nbors,
    h_params.max_rdonly * sizeof(int16_t), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(h_params.nbor_map, h_nbor_map, h_params.max_nbors * sizeof(int16_t),
    cudaMemcpyHostToDevice));

  cudaCheckError(cudaMemcpyToSymbol(d_params, &h_params, sizeof(GlobalConstants)));

  // printf("\n");
  // for (int i = 0; i < 3 * h_params.max_particles; i += 3) {
  //   printf("h_curr_particles[%d] = (%f, %f, %f)\n", i / 3, h_curr_particles[i], h_curr_particles[i + 1], h_curr_particles[i + 2]);
  // }
}

void solver_free(GlobalConstants& h_params)
{
  cudaCheckError(cudaFree(h_params.curr_particles));
  cudaCheckError(cudaFree(h_params.prev_particles));
  cudaCheckError(cudaFree(h_params.rdonly_nbors));
  cudaCheckError(cudaFree(h_params.nbor_map));
}
