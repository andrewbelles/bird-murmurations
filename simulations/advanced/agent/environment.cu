/*
 * environment.cu  Andrew Belles  Sept 29th, 2025 
 *
 * Implements the interface for parameter control and environment dynamic device 
 * functions defined in header file environment.cuh 
 *
 */

#include "environment.cuh"

#include <driver_types.h>
#include <iostream>  
#include <fstream>

__host__ cudaError_t 
read_environment_params(const char *path, EnvironmentParams *out)
{
  if ( !out || !path ) {
    std::cerr << "[ERROR] read_environment_params: nullptr argument\n";
    exit( 99 );
  }

  std::ifstream fptr(path);
  if ( !fptr.is_open() ) {
    std::cerr << "[ERROR] failed to open environment params file: " << path << '\n';
    exit( 99 ); 
  }

  // lambda helper to read variables from input file 
  EnvironmentParams params; 
  auto read = [&](float* var) {
    if ( !(fptr >> (*var)) ) {
      std::cerr << "[ERROR] failure while reading environment params file: "
                << path << '\n';
      exit( 99 );
    }
  };

  // read hyper params 
  read(&params.amplitude);
  read(&params.frequency);
  read(&params.noise);
  read(&params.origin.x);
  read(&params.origin.y); 
  read(&params.origin.z);
  
  // model level is optional, if read then set else default to 0 with warning 
  int model = 0; 
  if ( fptr >> model ) {
    if ( model < 0 || model > static_cast<int>(DynamicsLevel::Chaotic) ) {
      std::cerr << "[WARN] invalid dynamics level, defaulting to no dynamics\n"; 
      params.model = DynamicsLevel::None;
    } else {
      params.model = static_cast<DynamicsLevel>(model); 
    }
  } else {
    std::cerr << "[WARN] no model was provided, defaulting to no dynamics\n"; 
    params.model = DynamicsLevel::None; 
  }

  *out = params; 
  return cudaSuccess; 
}

__host__ cudaError_t 
read_simulation_params(const char *path, SimulationParams *out)
{
  if ( !out || !path ) {
    std::cerr << "[ERROR] read_simulation_params: nullptr argument\n";
    exit( 99 );
  }

  std::ifstream fptr(path);
  if ( !fptr.is_open() ) {
    std::cerr << "[ERROR] failed to open simulation params file: " << path << '\n';
    exit( 99 ); 
  }

  // lambda helper to read variables from input file 
  SimulationParams params; 
  auto read = [&](float* var) {
    if ( !(fptr >> (*var)) ) {
      std::cerr << "[ERROR] failure while reading simulation params file: "
                << path << '\n';
      exit( 99 );
    }
  };

  // read parameters 
  read(&params.dt);
  read(&params.min_dist);
  read(&params.w_separation); 
  read(&params.w_alignment);
  read(&params.w_cohesion);
  read(&params.max_accel);
  read(&params.max_vel);

  *out = params; 

  return cudaSuccess; 
}

/*
 * Zero map that has matching function prototype to subsequent environment functions 
 */ 
__device__ float3 
env_none(float3 pos, float time, const EnvironmentParams& params)
{
  return make_float3(0.0, 0.0, 0.0);
}

/*
 *
 */ 
__device__ float3 
env_vortex(float3 pos, float time, const EnvironmentParams& params)
{
  float3 r = make_float3(
    pos.x - params.origin.x,
    pos.y - params.origin.y,
    0.0
  );

  const float3 swirl = make_float3(-r.y, r.x, 0.0); 
  float mag = params.amplitude; 
  if ( params.frequency != 0.0 ) {
    mag *= sinf(params.frequency * time);
  }

  return make_float3(swirl.x * mag, swirl.y * mag, swirl.z * mag);
}

__device__ float3 
env_chaotic(float3 pos, float time, const EnvironmentParams &params)
{
  const float mag = params.amplitude; 
  auto turbulent = [&](float x) -> float {
    return x + (params.frequency * time) * mag;
  };

  return make_float3(turbulent(pos.x), turbulent(pos.y), turbulent(pos.z) );
}

__device__ float3 
environment_force(float3 pos, float time, const EnvironmentParams &params)
{
  switch ( params.model ) {
    case DynamicsLevel::Vortex: 
      return env_vortex(pos, time, params);
    case DynamicsLevel::Chaotic:
      return env_chaotic(pos, time, params);
    case DynamicsLevel::None: 
    default: 
      return env_none(pos, time, params);
  }
}
