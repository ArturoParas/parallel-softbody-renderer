#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <solver.hpp>
#include <circle.hpp>

#include "exclusiveScan.cu_inl"

#define THREADS_PER_BLOCK 512
#define GRAVITY_FORCE -0.98
#define MAX_CIRCLES 100
#define DAMPING_CONSTANT 0.98

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


//Assumptions:
//Only one block
//All particles have unit mass
//All springs have same k constant
//All springs have whole number resting lengths
//Sidenote the numbers are very magical
__device__ __inline__ void cuda_move_circle(float* circle_buffer, uint16_t* spring_buffer, 
                                            int num_circles, int num_springs, 
                                            float k_constant, float dt2,  
                                            int circle_struct_size, int spring_struct_size){

    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ float p_buffer[2*MAX_CIRCLES]; //cannot be variable length

    if(idx < num_circles*2){
        p_buffer[idx] = 0;
    }

    __syncthreads();

    /*------ Thread => Spring Endpoint ------*/

    if(idx < num_springs*2){

        //inline basically all of this later

        int endpoint_idx = static_cast<int>(floorf(3*idx/2));

        int p1_idx = spring_buffer[endpoint_idx]; 
        int p2_idx = spring_buffer[endpoint_idx + 1 - 2*(endpoint_idx%2)];
        
        float p1_x = circle_buffer[p1_idx * circle_struct_size + 4]; //
        float p2_x = circle_buffer[p2_idx * circle_struct_size + 4];

        float p1_y = circle_buffer[p1_idx * circle_struct_size + 5];
        float p2_y = circle_buffer[p2_idx * circle_struct_size + 5];

        float dx = p2_x - p1_x;
        float dy = p2_y - p1_y;

        float spring_length = sqrtf(dx*dx + dy*dy);

        float cos_theta = dx/spring_length;
        float sin_phi = dy/spring_length;

        uint16_t spring_rest_length = spring_buffer[static_cast<int>(3*floorf(idx/2)+2)];

        float signed_force_magnitude = k_constant * (spring_length - spring_rest_length)/spring_length;

        atomicAdd(&(p_buffer[2*endpoint_idx+0]), dt2 * abs(dx) * cos_theta * signed_force_magnitude);
        atomicAdd(&(p_buffer[2*endpoint_idx+1]), dt2 * abs(dy) * sin_phi * signed_force_magnitude);

        __syncthreads();
    }
    
    /*------------------------------*/
    idx = threadIdx.y * blockDim.x + threadIdx.x;
    /*------ Thread => Circle ------*/

    if(idx < num_circles){

        int circle_idx = idx * circle_struct_size;

        p_buffer[2*idx + 0] += circle_buffer[circle_idx + 2] + (circle_buffer[circle_idx + 2] - circle_buffer[circle_idx + 0]) * DAMPING_CONSTANT;
        p_buffer[2*idx + 1] += circle_buffer[circle_idx + 3] + (circle_buffer[circle_idx + 3] - circle_buffer[circle_idx + 1]) * DAMPING_CONSTANT + GRAVITY_FORCE * dt2;

        circle_buffer[circle_idx + 0] = circle_buffer[circle_idx + 2];
        circle_buffer[circle_idx + 1] = circle_buffer[circle_idx + 3];

        circle_buffer[circle_idx + 2] = p_buffer[2*idx + 0];
        circle_buffer[circle_idx + 3] = p_buffer[2*idx + 1];
        
        circle_buffer[circle_idx + 4] = p_buffer[2*idx + 0];
        circle_buffer[circle_idx + 5] = p_buffer[2*idx + 1];
        
        // printf("circle_idx %d :%.4f  %.4f\n",idx, circle_buffer[circle_idx + 2], circle_buffer[circle_idx +3]);
    }


    /*------------------------------*/


}

__device__ __inline__ void cuda_apply_border(float* circle_buffer, int width, int height, 
                                             int num_circles, int circle_radius,
                                             int circle_struct_size){

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    if(idx >= num_circles){
        return;
    }

    idx *= circle_struct_size;

    circle_buffer[idx + 2] = min(max(circle_buffer[idx + 2], circle_radius),width - circle_radius);
    circle_buffer[idx + 3] = min(max(circle_buffer[idx + 3], circle_radius),height - circle_radius);


}


__global__ void solver_update_cuda(float* circle_buffer, uint16_t* spring_buffer,
                                   int width, int height, 
                                   int num_circles, float circle_radius, 
                                   int num_springs, float k_constant,
                                   int intermediate_steps, float dt2, 
                                   int circle_struct_size, int spring_struct_size){

    for(int i = 0; i < intermediate_steps; i++){
        
        //Thread => circle
        cuda_move_circle(circle_buffer, spring_buffer, 
                         num_circles, num_springs, 
                         k_constant ,dt2, 
                         circle_struct_size, spring_struct_size);

        //cuda_resolve_collisions()

        cuda_apply_border(circle_buffer,width,height,num_circles,circle_radius,circle_struct_size);

    }

    

}



void solver_update(const softbody_sim::SolverInfo & solver_info, void* circles, void* springs){

    float* circle_buffer;
    uint16_t* spring_buffer;

    int circle_struct_size = 6; //magic number oops
    int spring_struct_size = 3;

    cudaMalloc(&circle_buffer, circle_struct_size * solver_info.num_circles * sizeof(float));
    cudaMemcpy(circle_buffer, circles, circle_struct_size * solver_info.num_circles * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&spring_buffer, spring_struct_size * solver_info.num_springs * sizeof(uint16_t));
    cudaMemcpy(spring_buffer, springs, spring_struct_size * solver_info.num_springs * sizeof(uint16_t), cudaMemcpyHostToDevice);

    solver_update_cuda<<<1,THREADS_PER_BLOCK>>>(circle_buffer, spring_buffer,
                                                solver_info.width, solver_info.height, 
                                                solver_info.num_circles,solver_info.circle_radius,
                                                solver_info.num_springs, solver_info.k_constant,
                                                solver_info.intermediate_steps, solver_info.dt2_intermediate, 
                                                circle_struct_size, spring_struct_size);

    cudaCheckError(cudaDeviceSynchronize());

    cudaMemcpy(circles, circle_buffer, circle_struct_size * solver_info.num_circles * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(circle_buffer);
    cudaFree(spring_buffer);

}

__global__ void device_test_spring_buffer(uint16_t *springs_device){

    printf("rmn: %u %u %u\n", (springs_device)[0], (springs_device)[1], (springs_device)[2]);
    
}

void initialize_springs(const softbody_sim::SolverInfo & solver_info, uint16_t *springs_host, uint16_t *springs_device){


    int spring_struct_size = 3; //another one :)

    cudaMalloc(&springs_device,solver_info.num_springs * sizeof(uint16_t) * spring_struct_size);
    cudaMemcpy(springs_device, springs_host, solver_info.num_springs * sizeof(uint16_t) * spring_struct_size, cudaMemcpyHostToDevice);

}

void host_test_spring_buffer(uint16_t *springs_device){

    device_test_spring_buffer<<<1,1>>>(springs_device);

    cudaCheckError(cudaDeviceSynchronize());

}


