#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <solver.hpp>
#include <circle.hpp>

#include "exclusiveScan.cu_inl"

#define NUM_BLOCKS 2

#define THREADS_PER_BLOCK 4
#define MAX_CIRCLES THREADS_PER_BLOCK * NUM_BLOCKS

#define MAX_RDONLY_PER_BLOCK 3
#define MAX_RDONLY MAX_RDONLY_PER_BLOCK * NUM_BLOCKS

#define MAX_NBORS_PER_CIRCLE 3
#define MAX_NBORS MAX_NBORS_PER_CIRCLE * MAX_CIRCLES

#define G -0.1
#define DAMP 0.98
#define DT 0.1
#define DT2 DT * DT
#define K 2.
#define RL 1
#define MASS 1

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

__device__ int rest_len;
__device__ int threads_per_block;
__device__ int num_blocks;
__device__ int max_pts;
__device__ int max_nbors_per_pt;
__device__ int max_nbors_per_block;
__device__ int max_nbors;
__device__ int max_rdonly_per_block;
__device__ int max_rdonly;
__device__ float dt2;
__device__ float g;
__device__ float k;

__device__ __inline__ void move()
{
  int t_idx = threadIdx.x;
  int b_idx = blockIdx.x;
  int g_off = b_idx * blockDim.x;
  int g_idx = g_off + t_idx;

  float3 prev_circle = g_prev_circles[g_idx];
  float3 curr_circle = g_curr_circles[g_idx];
  float3 next_circle = g_curr_circles[g_idx];

  next_circle.x += (curr_circle.x - prev_circle.x) * DAMP;
  next_circle.y += (curr_circle.y - prev_circle.y) * DAMP;
  next_circle.z += (curr_circle.z - prev_circle.z) * DAMP + G * DT2;

  for (int i = 0; i < MAX_NBORS_PER_CIRCLE; i++) {
    int n_idx = g_nbor_map[(g_off + i) * MAX_NBORS_PER_CIRCLE + t_idx];
    if (n_idx >= 0) {
      float3 nbor_circle = g_curr_circles[g_rdonly_nbors[g_off + n_idx]];

      float dx = nbor_circle.x - curr_circle.x;
      float dy = nbor_circle.y - curr_circle.y;
      float dz = nbor_circle.z - curr_circle.z;

      float dist = sqrtf(dx * dx + dy * dy + dz * dz);
      if (dist < 1e-16) {
        printf("TOO CLOSE\n");
      }

      float norm_a = K * (dist - RL) / dist / MASS;

      next_circle.x += norm_a * dx * DT2;
      next_circle.y += norm_a * dy * DT2;
      next_circle.z += norm_a * dz * DT2;
    }
  }
  __syncthreads();

  g_prev_circles[g_idx] = curr_circle;
  g_curr_circles[g_idx] = next_circle;
}

__global__ void trivial_kernel()
{
  move();
}

void solver_trivial(float* h_curr_circles)
{
  trivial_kernel<<<num_blocks, THREADS_PER_BLOCK>>>();
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpyFromSymbol(h_curr_circles, g_curr_circles, sizeof(float3) * max_pts));

  printf("\n");
  for (int i = 0; i < MAX_CIRCLES * 3; i += 3) {
    printf("h_curr_circles[%d] = (%f, %f, %f)\n", i / 3, h_curr_circles[i], h_curr_circles[i + 1], h_curr_circles[i + 2]);
  }
}

void solver_setup(const softbody_sim::SolverInfo& solver_info)
{
  cudaCheckError(cudaMemcpyToSymbol(rest_len, solver_info.rest_len, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(threads_per_block, solver_info.threads_per_block, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(num_blocks, solver_info.num_blocks, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(max_pts, solver_info.max_pts, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(max_nbors_per_pt, solver_info.max_nbors_per_pt, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(max_nbors_per_block, solver_info.max_nbors_per_block, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(max_nbors, solver_info.max_nbors, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(max_rdonly_per_block, solver_info.max_rdonly_per_block, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(max_rdonly, solver_info.max_rdonly, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(dt2, solver_info.dt2_intermediate, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(g, solver_info.gravity_acc, sizeof(int)));
  cudaCheckError(cudaMemcpyToSymbol(k, solver_info.k_constant, sizeof(int)));

  // cudaCheckError(cudaMemcpyToSymbol(g_curr_circles, h_curr_circles, sizeof(float3) * MAX_CIRCLES));
  // cudaCheckError(cudaMemcpyToSymbol(g_prev_circles, h_curr_circles, sizeof(float3) * MAX_CIRCLES));
  // cudaCheckError(cudaMemcpyToSymbol(g_rdonly_nbors, h_rdonly_nbors, sizeof(int) * MAX_RDONLY));
  // cudaCheckError(cudaMemcpyToSymbol(g_nbor_map, h_nbor_map, sizeof(int) * MAX_NBORS));

  // for (int i = 0; i < MAX_CIRCLES * 3; i += 3) {
  //   printf("h_curr_circles[%d] = (%f, %f, %f)\n", i / 3, h_curr_circles[i], h_curr_circles[i + 1], h_curr_circles[i + 2]);
  // }
  // printf("\n");
  // for (int i = 0; i < MAX_RDONLY; i++) {
  //   printf("h_rdonly_nbors[%d] = %d\n", i, h_rdonly_nbors[i]);
  // }
  // printf("\n");
  // for (int i = 0; i < MAX_NBORS; i++) {
  //   printf("h_nbor_map[%d] = %d\n", i, h_nbor_map[i]);
  // }
}
