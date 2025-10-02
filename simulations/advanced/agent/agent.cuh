/*
 * agent.cuh  Andrew Belles  Sept 29th 2025 
 *  
 * Interface for boid agents to interact both with surrounding agents 
 * but also make movement decisions based on environment dynamics 
 *
 */ 

#ifndef __AGENT_CUH 
#define __AGENT_CUH

#include <cuda_runtime.h> 

#include "environment.cuh"
#include "../network/mailbox.cuh"

namespace boids {

__host__ cudaError_t launch_compute(const float3* d_pos, const float3* d_vel, 
                                    const network::Mailbox& inbox, 
                                    const SimulationParams* sim_params, 
                                    const EnvironmentParams* env_params, 
                                    float3* d_accel, int N, uint64_t epoch,
                                    cudaStream_t stream = 0);

__host__ cudaError_t launch_integrate(float3* d_pos, float3* d_vel, 
                                      const float3* d_accel, int N,
                                      const SimulationParams* sim_params,
                                      cudaStream_t stream = 0);

__device__ float3 separation(const float3& avg_offset, const SimulationParams& params);

__device__ float3 alignment(const float3& avg_velocity, const SimulationParams& params, 
                            const float3& self_velocity);

__device__ float3 cohesion(const float3& avg_position, const SimulationParams& params, 
                           const float3& self_position);

__global__ void compute_boids_accel(const float3* d_pos, const float3* d_vel, 
                                    const network::Mailbox inbox,
                                    const SimulationParams* sim_params,
                                    const EnvironmentParams* env_params,
                                    float3* d_accel, int N, uint64_t epoch);

__global__ void integrate(float3* d_pos, float3* d_vel, const float3* d_accel, 
                          const SimulationParams* sim_params, int N);

} // end namespace boids 

#endif // __AGENT_CUH
