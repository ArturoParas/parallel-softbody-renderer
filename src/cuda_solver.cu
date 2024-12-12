#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <solver.hpp>
#include <circle.hpp>

#include "exclusiveScan.cu_inl"

#define CIRCLES_PER_BLOCK 39
#define NEIGHBORS_PER_CIRCLE 26
#define MAX_RDONLY_CIRCLE 8

#define THREADS_PER_BLOCK 512
#define GRAVITY_ACC -0.98
#define MAX_CIRCLES 100
#define DAMPING_CONSTANT 0.98
#define CIRCLE_RADIUS 2.5
#define K_CONSTANT 2
#define SPRING_REST_LENGTH 10
#define DT2 0.01
#define CIRCLE_MASS 1

#define WIDTH 100
#define DEPTH 100
#define HEIGHT 100


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


//Note that this implementation sacrifices a bit of correctness in exchange for a very efficient access pattern.
//That is, values that are being updated stay in the cache until they have been completely updated, and subsequent
//uses of the particle for updating neighboring particles will read the updated value.
__device__ __inline__ void update_circle_position_cuda_imprecise(float* shared_prev_circles, float* shared_curr_circles, uint16_t* nbors_buf){

    uint32_t tIdx = threadIdx.x;

    //Thread => Circle
    uint32_t cIdx = tIdx;
    while(cIdx < CIRCLES_PER_BLOCK){

        float circ_x = shared_curr_circles[3*cIdx + 0];
        float circ_y = shared_curr_circles[3*cIdx + 1];
        float circ_z = shared_curr_circles[3*cIdx + 2];

        float dist,move_amount,overlap; //For collision resolution
        float spring_length_xyz, spring_length_xy,spring_length_xz, x_contrib, y_contrib, z_contrib, signed_force_magnitude; //For spring application
        for(uint32_t i=0; i < NEIGHBORS_PER_CIRCLE; i++){

            /*-------- Collision Resolution --------*/

            float dx = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+0] - circ_x;
            float dy = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+1] - circ_y;
            float dz = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+2] - circ_z;

            dist = sqrtf(dx*dx + dy*dy + dz*dz);
            overlap = dist - 2*CIRCLE_RADIUS;

            if(dist-CIRCLE_RADIUS < 0){

                move_amount = overlap * 0.5f / dist;

                circ_x += dx * move_amount;
                circ_y += dy * move_amount;
                circ_z += dz * move_amount;

            }

            /*--------------------------------------*/

            /*---------- Force Application ---------*/

            if(i==0){
                circ_x += (circ_x - shared_prev_circles[3*cIdx + 0]) * DAMPING_CONSTANT;
                circ_y += (circ_y - shared_prev_circles[3*cIdx + 1]) * DAMPING_CONSTANT;
                circ_z += (circ_z - shared_prev_circles[3*cIdx + 2]) * DAMPING_CONSTANT + GRAVITY_ACC*DT2;
            }

            dx = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+0] - circ_x;
            dy = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+1] - circ_y;
            dz = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+2] - circ_z;

            spring_length_xyz = sqrtf(dx*dx + dy*dy + dz*dz);
            spring_length_xy = sqrtf(dx*dx + dy*dy);
            spring_length_xz = sqrtf(dx*dx + dz*dz);

            //I have no clue if these calculations are correct
            x_contrib = dx / spring_length_xy;
            y_contrib = dy / spring_length_xy;
            z_contrib = dz / spring_length_xz;

            signed_force_magnitude = K_CONSTANT * (spring_length_xyz - SPRING_REST_LENGTH); 

            circ_x += DT2 * abs(dx) * x_contrib * signed_force_magnitude;
            circ_y += DT2 * abs(dy) * y_contrib * signed_force_magnitude;
            circ_z += DT2 * abs(dz) * z_contrib * signed_force_magnitude;

            /*--------------------------------------*/

            /*------------ Apply Border ------------*/

            circ_x = min(max(circ_x, CIRCLE_RADIUS),WIDTH-CIRCLE_RADIUS);
            circ_y = min(max(circ_y, CIRCLE_RADIUS),DEPTH-CIRCLE_RADIUS);
            circ_z = min(max(circ_z, CIRCLE_RADIUS),HEIGHT-CIRCLE_RADIUS);

            /*--------------------------------------*/
        }

        __syncthreads();

        shared_curr_circles[3*cIdx + 0] = circ_x;
        shared_curr_circles[3*cIdx + 1] = circ_y;
        shared_curr_circles[3*cIdx + 2] = circ_z;

        __syncthreads();

        cIdx += blockDim.x;
    }

}


//Note that when CIRCLES_PER_BLOCK <= threads per block, this approach is completely precise in its calculations.
//The difference between this and the imprecise approach is that this method resolves all collisions before applying
//forces on the particles. 
__device__ __inline__ void update_circle_position_cuda_precise(float* shared_curr_circles, float* shared_prev_circles, uint16_t* nbors_buf){

    uint32_t tIdx = threadIdx.x;

    /*-------- Collision Resolution --------*/

    uint32_t cIdx = tIdx;
    while(cIdx < CIRCLES_PER_BLOCK){

        float dist,move_amount,overlap;
        for(uint32_t i=0; i < NEIGHBORS_PER_CIRCLE; i++){

            float dx = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+0] - shared_curr_circles[3*cIdx + 0];
            float dy = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+1] - shared_curr_circles[3*cIdx + 1];
            float dz = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+2] - shared_curr_circles[3*cIdx + 2];

            dist = sqrtf(dx*dx + dy*dy + dz*dz);
            overlap = dist - 2*CIRCLE_RADIUS;

            if(dist-CIRCLE_RADIUS < 0){

                move_amount = overlap * 0.5f / dist;

                shared_curr_circles[3*cIdx + 0] += dx * move_amount;
                shared_curr_circles[3*cIdx + 1] += dy * move_amount;
                shared_curr_circles[3*cIdx + 2] += dz * move_amount;

            }

        }

        cIdx += blockDim.x;
    }

    /*--------------------------------------*/

    //Could insert syncthreads here

    /*---------- Force Application ---------*/

    cIdx = tIdx;
    while(cIdx < CIRCLES_PER_BLOCK){

        float orig_x = shared_curr_circles[3*cIdx + 0];
        float orig_y = shared_curr_circles[3*cIdx + 1];
        float orig_z = shared_curr_circles[3*cIdx + 2];

        float circ_x = orig_x;
        float circ_y = orig_y;
        float circ_z = orig_z;

        circ_x += (circ_x - shared_prev_circles[3*cIdx + 0]) * DAMPING_CONSTANT;
        circ_y += (circ_y - shared_prev_circles[3*cIdx + 1]) * DAMPING_CONSTANT;
        circ_z += (circ_z - shared_prev_circles[3*cIdx + 2]) * DAMPING_CONSTANT + GRAVITY_ACC*DT2;

        float spring_length_xyz, spring_length_xy,spring_length_xz, x_contrib, y_contrib, z_contrib, signed_force_magnitude; 
        for(uint32_t i=0; i < NEIGHBORS_PER_CIRCLE; i++){

            float dx = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+0] - orig_x;
            float dy = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+1] - orig_y;
            float dz = shared_curr_circles[nbors_buf[cIdx + CIRCLES_PER_BLOCK*i]+2] - orig_z;

            spring_length_xyz = sqrtf(dx*dx + dy*dy + dz*dz);
            spring_length_xy = sqrtf(dx*dx + dy*dy);
            spring_length_xz = sqrtf(dx*dx + dz*dz);

            //I have no clue if these three calculations are correct
            x_contrib = dx / spring_length_xy;
            y_contrib = dy / spring_length_xy;
            z_contrib = dz / spring_length_xz;

            signed_force_magnitude = K_CONSTANT * (spring_length_xyz - SPRING_REST_LENGTH); 

            circ_x += DT2 * abs(dx) * x_contrib * signed_force_magnitude / CIRCLE_MASS;
            circ_y += DT2 * abs(dy) * y_contrib * signed_force_magnitude / CIRCLE_MASS;
            circ_z += DT2 * abs(dz) * z_contrib * signed_force_magnitude / CIRCLE_MASS;

        }

        /*--------------------------------------*/

        /*---------- Border Application ---------*/

        shared_curr_circles[3*cIdx + 0] = min(max(circ_x, CIRCLE_RADIUS),WIDTH-CIRCLE_RADIUS);
        shared_curr_circles[3*cIdx + 1] = min(max(circ_y, CIRCLE_RADIUS),DEPTH-CIRCLE_RADIUS);
        shared_curr_circles[3*cIdx + 2] = min(max(circ_z, CIRCLE_RADIUS),HEIGHT-CIRCLE_RADIUS);
        
        /*--------------------------------------*/

        cIdx += blockDim.x;
    }

}

void __global__ solver_update_cuda(float* device_prev_circles, float* device_curr_circles, 
                              uint16_t* device_neighbor_indices, uint16_t* device_neighbor_map,
                              uint32_t intermediate_steps, uint32_t num_circles){

    //Grid will be organized in a linear fashion i.e 0,1,...,n | 0,1,...,n | ... | 0,1,...,n
    uint32_t bIdx = blockIdx.x; //Block index
    uint32_t tIdx = threadIdx.x; //Thread index
    uint32_t gIdx = blockDim.x * bIdx + threadIdx.x; //Global thread index

    __shared__ float shared_prev_circles[3*CIRCLES_PER_BLOCK];
    __shared__ float shared_curr_circles[3*(CIRCLES_PER_BLOCK + NEIGHBORS_PER_CIRCLE)];
    __shared__ uint16_t nbors_buf[NEIGHBORS_PER_CIRCLE*CIRCLES_PER_BLOCK];

    //Populate update circles
    uint32_t cIdx = tIdx;
    while(cIdx < 3*CIRCLES_PER_BLOCK){

        shared_prev_circles[cIdx] = device_prev_circles[bIdx*3*CIRCLES_PER_BLOCK + cIdx];
        shared_curr_circles[cIdx] = device_curr_circles[bIdx*3*CIRCLES_PER_BLOCK + cIdx];
        cIdx += blockDim.x;
    }

    //Populate rdonly circles
    cIdx = tIdx;
    while(cIdx < 3*MAX_RDONLY_CIRCLE){
        
        shared_curr_circles[cIdx + 3*CIRCLES_PER_BLOCK] = device_curr_circles[3*device_neighbor_indices[cIdx/3] + (cIdx%3)];
        cIdx += blockDim.x;
    }

    //Populate neighbor indices
    cIdx=tIdx;
    while(cIdx < NEIGHBORS_PER_CIRCLE*CIRCLES_PER_BLOCK){

        nbors_buf[cIdx] = device_neighbor_map[bIdx*NEIGHBORS_PER_CIRCLE*CIRCLES_PER_BLOCK + cIdx];
        cIdx += blockDim.x;
    }

    __syncthreads();


    for(uint32_t i = 0; i < intermediate_steps; i++){

        // update_circle_position_cuda_imprecise(shared_curr_circles, shared_prev_circles, nbors_buf);
        update_circle_position_cuda_precise(shared_curr_circles, shared_prev_circles, nbors_buf);
        __syncthreads();
    }

    cIdx=tIdx;
    while(cIdx < 3*CIRCLES_PER_BLOCK){

        device_prev_circles[3*CIRCLES_PER_BLOCK*bIdx + cIdx] = shared_prev_circles[cIdx];
        device_curr_circles[3*CIRCLES_PER_BLOCK*bIdx + cIdx] = shared_curr_circles[cIdx];
        cIdx += blockDim.x;
    }

}
