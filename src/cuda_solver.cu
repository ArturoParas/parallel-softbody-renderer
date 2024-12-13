#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <solver.hpp>
#include <circle.hpp>

#include "exclusiveScan.cu_inl"

#define NEIGHBORS_PER_CIRCLE 26
#define MAX_RDONLY_CIRCLES 380


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

/** TODO: Change gravity_force to gravity_accel */

__device__ __inline__ void update_circle_position_cuda_imprecise(float* shared_curr_circles, float* shared_prev_circles, uint16_t* nbors_buf, 
                                                                 float width, float depth, float height, 
                                                                 float circle_radius, float circle_mass, float k_constant, float spring_rest_length,
                                                                 float damping_constant, float gravity_force, float dt2){

    uint32_t tIdx = threadIdx.x;

    //Thread => Circle

    float circ_x = shared_curr_circles[3*tIdx+ 0];
    float circ_y = shared_curr_circles[3*tIdx+ 1];
    float circ_z = shared_curr_circles[3*tIdx+ 2];

    circ_x += (circ_x - shared_prev_circles[3*tIdx+ 0]) * damping_constant;
    circ_y += (circ_y - shared_prev_circles[3*tIdx+ 1]) * damping_constant;
    circ_z += (circ_z - shared_prev_circles[3*tIdx+ 2]) * damping_constant + gravity_force*dt2/circle_mass;

    float dist,move_amount,overlap; //For collision resolution
    float spring_length, dx, dy, dz, x_dir, y_dir, z_dir, signed_force_magnitude; //For spring application
    for(uint32_t i=0; i < NEIGHBORS_PER_CIRCLE; i++){

        /*---------- Force Application ---------*/

        dx = shared_curr_circles[nbors_buf[tIdx+ threads_per_block*i]+0] - circ_x;
        dy = shared_curr_circles[nbors_buf[tIdx+ threads_per_block*i]+1] - circ_y;
        dz = shared_curr_circles[nbors_buf[tIdx+ threads_per_block*i]+2] - circ_z;

        spring_length = sqrtf(dx*dx + dy*dy + dz*dz);

        x_dir = dx / spring_length;
        y_dir = dy / spring_length;
        z_dir = dz / spring_length;

        signed_force_magnitude = k_constant * (spring_length - spring_rest_length); 

        circ_x += dt2 * x_dir * signed_force_magnitude / circle_mass;
        circ_y += dt2 * y_dir * signed_force_magnitude / circle_mass;
        circ_z += dt2 * z_dir * signed_force_magnitude / circle_mass;

        shared_curr_circles[3*tIdx+ 0] = circ_x;
        shared_curr_circles[3*tIdx+ 1] = circ_y;
        shared_curr_circles[3*tIdx+ 2] = circ_z;

        /*--------------------------------------*/
        //Could insert a syncthreads here
        /*-------- Collision Resolution --------*/

        dx = shared_curr_circles[nbors_buf[tIdx+ threads_per_block*i]+0] - circ_x;
        dy = shared_curr_circles[nbors_buf[tIdx+ threads_per_block*i]+1] - circ_y;
        dz = shared_curr_circles[nbors_buf[tIdx+ threads_per_block*i]+2] - circ_z;

        dist = sqrtf(dx*dx + dy*dy + dz*dz);
        overlap = dist - 2*circle_radius;

        if(dist-circle_radius < 0){

            move_amount = overlap * 0.5f / dist;

            circ_x += dx * move_amount;
            circ_y += dy * move_amount;
            circ_z += dz * move_amount;

        }

        /*--------------------------------------*/

        /*------------ Apply Border ------------*/

        circ_x = min(max(circ_x, circle_radius),width -circle_radius);
        circ_y = min(max(circ_y, circle_radius),depth -circle_radius);
        circ_z = min(max(circ_z, circle_radius),height -circle_radius);

        /*--------------------------------------*/
    }

    __syncthreads();

    shared_curr_circles[3*tIdx+ 0] = circ_x;
    shared_curr_circles[3*tIdx+ 1] = circ_y;
    shared_curr_circles[3*tIdx + 2] = circ_z;




}


__device__ __inline__ void update_circle_position_cuda_precise(float* shared_curr_circles, float* shared_prev_circles, uint16_t* nbors_buf,
                                                               float width, float depth, float height, 
                                                               float circle_radius, float circle_mass, float k_constant, float spring_rest_length,
                                                               float damping_constant, float gravity_force, float dt2){

    uint32_t tIdx = threadIdx.x;

    /*---------- Force Application ---------*/

    float orig_x = shared_curr_circles[3*tIdx + 0];
    float orig_y = shared_curr_circles[3*tIdx + 1];
    float orig_z = shared_curr_circles[3*tIdx + 2];

    float circ_x = orig_x;
    float circ_y = orig_y;
    float circ_z = orig_z;

    circ_x += (circ_x - shared_prev_circles[3*tIdx + 0]) * damping_constant;
    circ_y += (circ_y - shared_prev_circles[3*tIdx + 1]) * damping_constant;
    circ_z += (circ_z - shared_prev_circles[3*tIdx + 2]) * damping_constant + gravity_force*dt2/circle_mass;

    float spring_length, dx, dy, dz, x_dir, y_dir, z_dir, signed_force_magnitude; 
    uint16_t nIdx;
    for(uint32_t i=0; i < NEIGHBORS_PER_CIRCLE; i++){

        nIdx = nbors_buf[tIdx + threads_per_block*i];
        if(nIdx >= 0){
            dx = shared_curr_circles[3*nIdx+0] - orig_x;
            dy = shared_curr_circles[3*nIdx+1] - orig_y;
            dz = shared_curr_circles[3*nIdx+2] - orig_z;

            spring_length = sqrtf(dx*dx + dy*dy + dz*dz);

            x_dir = dx / spring_length;
            y_dir = dy / spring_length;
            z_dir = dz / spring_length;

            signed_force_magnitude = k_constant * (spring_length - spring_rest_length); 

            circ_x += dt2 * x_dir * signed_force_magnitude / circle_mass;
            circ_y += dt2 * y_dir * signed_force_magnitude / circle_mass;
            circ_z += dt2 * z_dir * signed_force_magnitude / circle_mass;
        }
        //mayhaps syncthreads

    }

    /*--------------------------------------*/

    __syncthreads();

    shared_prev_circles[3*tIdx + 0] = shared_curr_circles[3*tIdx + 0];
    shared_prev_circles[3*tIdx + 1] = shared_curr_circles[3*tIdx + 1];
    shared_prev_circles[3*tIdx + 2] = shared_curr_circles[3*tIdx + 2];

    shared_curr_circles[3*tIdx + 0] = circ_x;
    shared_curr_circles[3*tIdx + 1] = circ_y;
    shared_curr_circles[3*tIdx + 2] = circ_z;

    __syncthreads();

    /*-------- Collision Resolution --------*/

    orig_x = circ_x; 
    orig_y = circ_y; 
    orig_z = circ_z; 

    float dist,move_amount,overlap;
    for(uint32_t i=0; i < NEIGHBORS_PER_CIRCLE; i++){

        nIdx = nbors_buf[tIdx + threads_per_block*i];
        if(nIdx >= 0){

            dx = shared_curr_circles[nIdx+0] - orig_x;
            dy = shared_curr_circles[nIdx+1] - orig_y;
            dz = shared_curr_circles[nIdx+2] - orig_z;

            dist = sqrtf(dx*dx + dy*dy + dz*dz);
            overlap = dist - 2*circle_radius;

            if(overlap < 0){

                move_amount = overlap * 0.5f / dist;

                circ_x += dx * move_amount;
                circ_y += dy * move_amount;
                circ_z += dz * move_amount;

            }

        }

        //mayhaps syncthreads
    }
    
    /*--------------------------------------*/
    __syncthreads();
    /*---------- Border Application ---------*/

    shared_curr_circles[3*tIdx + 0] = min(max(circ_x, -width/2  + circle_radius), width/2  - circle_radius);
    shared_curr_circles[3*tIdx + 1] = min(max(circ_y, -depth/2  + circle_radius), depth/2  - circle_radius);
    shared_curr_circles[3*tIdx + 2] = min(max(circ_z, -height/2 + circle_radius), height/2 - circle_radius);


    
    /*--------------------------------------*/
    
}

__global__ void solver_cuda(float* device_curr_circles, float* device_prev_circles, 
                            uint16_t* device_neighbor_indices, uint16_t* device_neighbor_map,
                            float width, float depth, float height, 
                            float circle_radius, float circle_mass, 
                            float k_constant, float spring_rest_length,
                            float damping_constant, float gravity_force, float dt2){

    //Grid will be organized in a linear fashion i.e 0,1,...,n | 0,1,...,n | ... | 0,1,...,n
    uint32_t bIdx = blockIdx.x; //Block index
    uint32_t tIdx = threadIdx.x; //Thread index

    __shared__ float shared_prev_circles[3*threads_per_block];
    __shared__ float shared_curr_circles[3*(threads_per_block + MAX_RDONLY_CIRCLES)];
    __shared__ uint16_t nbors_buf[NEIGHBORS_PER_CIRCLE*threads_per_block];

    //Populate update circles
    uint32_t cIdx = tIdx;
    while(cIdx < 3*threads_per_block){

        shared_prev_circles[cIdx] = device_prev_circles[bIdx*3*threads_per_block + cIdx];
        shared_curr_circles[cIdx] = device_curr_circles[bIdx*3*threads_per_block + cIdx];
        cIdx += threads_per_block;
    }

    //Populate rdonly circles
    cIdx = tIdx;
    while(cIdx < 3*MAX_RDONLY_CIRCLES){
        
        shared_curr_circles[cIdx + 3*threads_per_block] = device_curr_circles[3*device_neighbor_indices[bIdx*MAX_RDONLY_CIRCLES + cIdx/3] + (cIdx%3)]; //device_neighbor_indices
        cIdx += threads_per_block;
    }

    //Populate neighbor indices
    cIdx = tIdx;
    while(cIdx < NEIGHBORS_PER_CIRCLE*threads_per_block){

        nbors_buf[cIdx] = device_neighbor_map[bIdx*NEIGHBORS_PER_CIRCLE*threads_per_block + cIdx];
        cIdx += threads_per_block;
    }

    __syncthreads();

    for(uint32_t i = 0; i < intermediate_steps; i++){

        // update_circle_position_cuda_precise(shared_curr_circles, shared_prev_circles, nbors_buf, ); //TODOOO
        __syncthreads();
    }

    cIdx = tIdx;
    while(cIdx < 3*threads_per_block){

        device_prev_circles[3*threads_per_block*bIdx + cIdx] = shared_prev_circles[cIdx];
        device_curr_circles[3*threads_per_block*bIdx + cIdx] = shared_curr_circles[cIdx];
        cIdx += threads_per_block;
    }

}


void solver_update(float* host_curr_circles, float* device_curr_circles, float* device_prev_circles, 
                   uint16_t* device_neighbor_indices, uint16_t* device_neighbor_map, 
                   softbody_sim::SolverInfo & solver_info){

    

    solver_cuda<<<solver_info.num_blocks, solver_info.threads_per_block>>>(device_curr_circles, device_prev_circles,
                                                                           device_neighbor_indices, device_neighbor_map,
                                                                           solver_info.width, solver_info.depth, solver_info.height,
                                                                           solver_info.circle_radius, solver_info.circle_mass,
                                                                           solver_info.k_constant, solver_info.spring_resting_length,
                                                                           solver_info.damping_constant, solver_info.gravity_force, 
                                                                           solver_info.intermediate_steps, solver_info.dt2_intermediate);

    cudaMemcpy(host_curr_circles, device_curr_circles, 3*solver_info.num_blocks*solver_info.threads_per_block,cudaMemcpyDeviceToHost);

}


