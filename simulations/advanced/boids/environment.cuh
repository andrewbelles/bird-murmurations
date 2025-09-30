/*
 * environment.cuh  Andrew Belles  Sept 29th, 2025 
 *
 * Module that defines parameters, hyperparameters, and environmental dynamics 
 * of the system that simulation uses with an emphasis on operability and flexibility  
 *
 */ 

#ifndef __ENVIRONMENT_CUH 
#define __ENVIRONMENT_CUH

#include <cuda_runtime.h> 
#include <cstdint> 

/*
 * Enum defining the increasing levels of dynamics for system to enforce 
 */ 
enum class DynamicsLevel : uint8_t {
  None    = 0, // no external forces 
  Vortex  = 1, // swirling vector field 
  Chaotic = 2  // Turbulent flow 
};

struct EnvironmentParams {
  DynamicsLevel model = DynamicsLevel::None; 
  float amplitude = 0.0;  // strength of field dynamics 
  float frequency = 0.0;  // oscillation of temp-dependent dynamics 
  float noise     = 0.0;  // controls stochastic element to dynamics 
  float3 origin;
};

struct SimulationParams {
  float dt           = 1e-3; 
  float min_dist     = 0.3;  // target minimum separation between boids 
  float w_separation = 1.0;  // weight of separation force  
  float w_alignment  = 1.0;  // weight of alignment force 
  float w_cohesion   = 1.0;  // weight of cohesion force 
  float max_accel    = 5.0;  // maximal acceleration 
  float max_vel      = 5.0;  // maximal velocity 
};

__host__ cudaError_t read_environment_params(const char* path, EnvironmentParams* out);

__host__ cudaError_t read_simulation_params(const char* path, SimulationParams* out);

template <typename T> __host__ cudaError_t copy_params(const T* params, T** d_out);

template <typename T> __host__ void destroy_params(T* d_params); 

// clamp overload for float3, aggressive inlining 
__host__ __device__ __forceinline__ float3 
clamp(const float3& v, float lo, float hi)
{
  float3 r; 
  r.x = (v.x > hi ? hi : (v.x < lo ? lo : v.x));
  r.y = (v.y > hi ? hi : (v.y < lo ? lo : v.y));
  r.z = (v.z > hi ? hi : (v.z < lo ? lo : v.z));
  return r; 
}

__device__ float3 env_none(float3 pos, float time, const EnvironmentParams& params);
__device__ float3 env_vortex(float3 pos, float time, const EnvironmentParams& params);
__device__ float3 env_chaotic(float3 pos, float time, const EnvironmentParams& params);

__device__ float3 environment_force(float3 pos, float time,
                                    const EnvironmentParams& params);

#endif // __ENVIRONMENT_CUH
