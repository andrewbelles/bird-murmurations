/*
 *
 * agent.cu  Andrew Belles  Sept 29th, 2025 
 *
 * Implementation of main boids kernels for boids agents 
 *
 */

#include "agent.cuh"
#include "environment.cuh"

#include <cuda_runtime.h> 
#include <cstdio> 

namespace boids {

__constant__ float3 zero{0.0, 0.0, 0.0};

__host__ cudaError_t 
launch_compute(const float3 *d_pos, const float3 *d_vel, 
              const network::Mailbox &inbox, const SimulationParams *sim_params, 
              const EnvironmentParams *env_params, float3 *d_accel, int N,
              uint64_t epoch, cudaStream_t stream)
{
  int block = 256; 
  int grid = (N + block - 1) / block;
  compute_boids_accel<<<grid, block, 0, stream>>>(
    d_pos, d_vel, inbox, sim_params, env_params, d_accel, N, epoch
  );
  return cudaGetLastError();
}

__host__ cudaError_t 
launch_integrate(float3 *d_pos, float3 *d_vel, const float3 *d_accel, int N,
                 const SimulationParams *sim_params, cudaStream_t stream)
{
  int block = 256; 
  int grid = (N + block - 1) / block;
  integrate<<<grid, block, 0, stream>>>(d_pos, d_vel, d_accel, sim_params, N);
  return cudaGetLastError();
}

/************ float3 overloads ****************************/

__device__ __host__ inline float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3& operator+=(float3& a, const float3& b)
{
  a.x += b.x;
  a.y += b.y; 
  a.z += b.z; 
  return a; 
}

__device__ __host__ inline float3& operator-=(float3& a, const float3& b)
{
  a.x -= b.x;
  a.y -= b.y; 
  a.z -= b.z; 
  return a; 
}

__device__ __host__ inline float3 operator*(const float& c, const float3& b)
{
  return make_float3(c * b.x, c * b.y, c * b.z);
}

__device__ float3 
separation(const float3 &avg_offset, const SimulationParams &params)
{
  return params.w_separation * avg_offset; 
}

__device__ float3 
alignment(const float3 &avg_velocity, const SimulationParams &params, 
          const float3 &self_velocity)
{
  return params.w_alignment * (avg_velocity - self_velocity); 
}

__device__ float3 
cohesion(const float3 &avg_position, const SimulationParams &params, 
         const float3 &self_position) 
{
  return params.w_cohesion * (avg_position - self_position); 
}

__global__ void 
compute_boids_accel(const float3 *d_pos, const float3 *d_vel, 
                    const network::Mailbox inbox, const SimulationParams *sim_params, 
                    const EnvironmentParams *env_params, float3 *d_accel, int N, 
                    uint64_t epoch)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( idx >= N ) {
    return; 
  }

  // pool parameters into registers 
  const SimulationParams sp  = *sim_params; 
  const EnvironmentParams ep = *env_params;
  float3 offset_sum = zero, velocity_sum = zero, position_sum = zero; 
  int neighbor_count = 0;
  float3 self_pos = d_pos[idx], self_vel = d_vel[idx];
  
  auto accumulate = [&] __device__ (int, const network::Packet& packet) {
    if ( packet.agent_id == idx ) {
      return; 
    } 

    offset_sum   += self_pos - packet.x; 
    velocity_sum += packet.v;  
    position_sum += packet.x; 
    neighbor_count++; 
  };

  network::drain_iterator(inbox, idx, 0x7fffffff, accumulate);
  float3 sep = zero, align = zero, coh = zero; 

  if ( neighbor_count > 0 ) {
    const float inv = 1.0 / static_cast<float>(neighbor_count);
    float3 avg_offset, avg_vel, avg_pos; 
    avg_offset = inv * offset_sum; 
    avg_vel    = inv * velocity_sum; 
    avg_pos    = inv * position_sum; 

    sep   = separation(avg_offset, sp); 
    align = alignment(avg_vel, sp, self_vel);
    coh   = cohesion(avg_pos, sp, self_pos);
  }

  float time = static_cast<float>(epoch) * sp.dt; 
  float3 env = environment_force(self_pos, time, ep); 
  float3 accel = clamp(sep + align + coh + env, -sp.max_accel, sp.max_accel);

  d_accel[idx] = accel;
}

__global__ void 
integrate(float3* d_pos, float3* d_vel, const float3* d_accel, 
          const SimulationParams* sim_params, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( idx >= N ) {
    return; 
  }

  const SimulationParams sp = *sim_params;
  const float extent = sp.world_extent; 
  float3 vel = d_vel[idx], acc = d_accel[idx], pos = d_pos[idx];
  
  float3 tile = make_float3(
    floorf((pos.x + extent) / (2.0 * extent)), 
    floorf((pos.y + extent) / (2.0 * extent)),
    floorf((pos.z + extent) / (2.0 * extent))
  );

  vel += sp.dt * acc; 
  vel = clamp(vel, -sp.max_vel, sp.max_vel);
  d_vel[idx] = vel; 

  pos += sp.dt * vel; 
  pos -= (2.0 * extent) * tile;
  d_pos[idx] = pos; 
}

} // end namespace boids 
