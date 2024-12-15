#include <stdio.h>
#include <iostream>

#include "../include/cuda_solver.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define min(a,b) (a < b ? a : b)
#define max(a,b) (a > b ? a : b)
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

__constant__ GlobalConstants d_params;

__device__ __inline__ void move(const int t_idx, const int b_idx, const int d_off, const int d_idx)
{
  float3 prev_particle = d_params.prev_particles[d_idx];
  float3 curr_particle = d_params.curr_particles[d_idx];
  float3 next_particle = curr_particle;

  // Move according to current velocity
  next_particle.x += (curr_particle.x - prev_particle.x) * d_params.spring_damp;
  next_particle.y += (curr_particle.y - prev_particle.y) * d_params.spring_damp;
  next_particle.z += (curr_particle.z - prev_particle.z) * d_params.spring_damp;

  // Move according to current acceleration
  for (int nbor = 0; nbor < d_params.max_nbors_per_particle; nbor++) {
    int n_key = d_params.nbor_map[d_off * d_params.max_nbors_per_particle + nbor * d_params.particles_per_block + t_idx];
    if (n_key >= 0) {
      int n_idx = 0;
      if (n_key < d_params.particles_per_block) {
        n_idx = d_off + n_key;
      } else {
        n_idx = d_params.rdonly_nbors[b_idx * d_params.max_rdonly_per_block + n_key - d_params.particles_per_block];
      }
      float3 nbor_particle = d_params.curr_particles[n_idx];

      float dx = nbor_particle.x - curr_particle.x;
      float dy = nbor_particle.y - curr_particle.y;
      float dz = nbor_particle.z - curr_particle.z;

      float dist = sqrtf(dx * dx + dy * dy + dz * dz);
      // DEBUG
      // if (dist < 1e-16) {
      //   printf("TOO CLOSE\n");
      // }

      float norm_acc = d_params.spring_k * (dist - d_params.spring_rest_len)
        / (dist * d_params.particle_mass);
      // printf("acc %f, k %f, dist %f, rest len %d, mass %f\n", norm_acc, d_params.spring_k, dist, d_params.spring_rest_len, d_params.particle_mass);

      next_particle.x += norm_acc * dx * d_params.dt2_intermediate;
      next_particle.y += norm_acc * dy * d_params.dt2_intermediate;
      next_particle.z += (norm_acc * dz + d_params.g) * d_params.dt2_intermediate;
      // printf("change in x %f, dx %f, dt2 intermediate %f\n", norm_acc * dx * d_params.dt2_intermediate, dx, d_params.dt2_intermediate);
      // printf("change in y %f, dy %f, dt2 intermediate %f\n", norm_acc * dy * d_params.dt2_intermediate, dy, d_params.dt2_intermediate);
      // printf("change in z %f, dz %f, dt2 intermediate %f\n", norm_acc * dz * d_params.dt2_intermediate, dz, d_params.dt2_intermediate);
    }
  }
  __syncthreads();

  d_params.prev_particles[d_idx] = curr_particle;
  d_params.curr_particles[d_idx] = next_particle;
}

__global__ void update_kernel()
{
  int t_idx = threadIdx.x;
  int b_idx = blockIdx.x;
  int d_off = b_idx * blockDim.x;
  int d_idx = d_off + t_idx;

  move(t_idx, b_idx, d_off, d_idx);
}

void solver_update(GlobalConstants& h_params, float* h_curr_particles)
{
  update_kernel<<<h_params.num_blocks, h_params.particles_per_block>>>();
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(h_curr_particles, h_params.curr_particles, h_params.max_particles * sizeof(float3), cudaMemcpyDeviceToHost));

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
  
  cudaCheckError(cudaMemcpy(h_params.curr_particles, h_curr_particles, h_params.max_particles * sizeof(float3), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(h_params.prev_particles, h_curr_particles, h_params.max_particles * sizeof(float3), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(h_params.rdonly_nbors, h_rdonly_nbors, h_params.max_rdonly * sizeof(int16_t), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(h_params.nbor_map, h_nbor_map, h_params.max_nbors * sizeof(int16_t), cudaMemcpyHostToDevice));

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
