/*
 * simulation.cu  Andrew Belles  Oct 1st, 2025 
 *
 * Implements interface to initialize and maintain simulation 
 *
 *
 */ 

#include "simulation.cuh"
#include "network/grid.cuh"
#include "network/mailbox.cuh"
#include "network/communication.cuh" 
#include "logger/logger.cuh"
#include "agent/environment.cuh"
#include "agent/agent.cuh"

#include <cuda_runtime.h> 
#include <iostream> 
#include <cfloat> 

namespace {
static inline float3 get_max_bound(const float3* h_pos, int N);
static inline float3& operator*=(float3& a, const float& c);
}


namespace sim {

/*
 * Parent Initializer of Simulation 
 *
 * We assume: 
 *   Caller will exit nonzero on return of status != cudaSuccess, thus we do not worry 
 *   about gracefully freeing before returning 
 */ 
cudaError_t 
create(Simulation* sim, const float3* h_pos, const float* h_vel, 
       int agent_count, int bufr_size, const std::string& logger_path,
       const SimulationParams* sim_params, 
       const EnvironmentParams* env_params)
{
  cudaError_t status; 
  size_t bytes(sizeof(float3) * static_cast<size_t>(agent_count));

  if ( !sim || !h_pos || !h_vel || !sim_params || !env_params ) {
    std::cerr << "[ERROR] recieved null arguments\n";
    return cudaErrorInvalidValue; 
  }

  sim->agent_count  = agent_count; 
  sim->bufr_size    = bufr_size; 
  sim->d_pos        = nullptr; 
  sim->d_vel        = nullptr; 
  sim->d_accel      = nullptr; 
  sim->d_sim_params = nullptr; 
  sim->d_env_params = nullptr; 

  // state mallocs 

  if ( (status = cudaMalloc(&sim->d_pos, bytes)) != cudaSuccess ) {
    std::cerr << "[ERROR] d_pos malloc choked\n";
    return status; 
  }

  if ( (status = cudaMalloc(&sim->d_vel, bytes)) != cudaSuccess ) {
    std::cerr << "[ERROR] d_vel malloc choked\n";
    return status; 
  }

  if ( (status = cudaMalloc(&sim->d_accel, bytes)) != cudaSuccess ) {
    std::cerr << "[ERROR] d_accel malloc choked\n";
    return status; 
  }

  // copies 

  if ( (status = cudaMemcpy(sim->d_pos, h_pos, bytes, cudaMemcpyHostToDevice)) 
      != cudaSuccess ) {
    std::cerr << "[ERROR] d_pos memcpy choked\n"; 
    return status; 
  }

  if ( (status = cudaMemcpy(sim->d_vel, h_vel, bytes, cudaMemcpyHostToDevice)) 
      != cudaSuccess ) {
    std::cerr << "[ERROR] d_vel memcpy choked\n"; 
    return status; 
  }

  // get parameters 

  if ( (status = copy_params<SimulationParams>(sim_params, &sim->d_sim_params)) 
      != cudaSuccess ) {
    std::cerr << "[ERROR] failed to copy simulation parameters to device\n"; 
    return status; 
  }

  if ( (status = copy_params<EnvironmentParams>(env_params, &sim->d_env_params)) 
      != cudaSuccess ) {
    std::cerr << "[ERROR] failed to copy environment parameters to device\n"; 
    return status; 
  }

  float cell_size = (sim_params->min_dist <= 0.0) ? 1.0 : sim_params->min_dist; 
  float3 max_bound = get_max_bound(h_pos, agent_count);
  if ( (status = sim->grid.create(max_bound, cell_size, agent_count)) 
      != cudaSuccess ) {
    std::cerr << "[ERROR] failed to create grid\n"; 
    return status; 
  }

  if ( (status = sim->inbox.create(agent_count, bufr_size)) != cudaSuccess ) {
    std::cerr << "[ERROR] failed to initialize ring buffers\n"; 
    return status; 
  } 

  const int batch_size = 1;
  if ( (status = sim->logger.create(agent_count, batch_size, logger_path.c_str())) 
      != cudaSuccess ) {
    std::cerr << "[ERROR] failed to initialize data logger\n"; 
  }
  
  return cudaSuccess; 
}

cudaError_t 
destroy(Simulation* sim) 
{
  sim->logger.destroy(); // logger should be closed first, flushes remaining epochs 

  sim->inbox.destroy();
  sim->grid.destroy();

  if ( sim->d_pos ) {
      cudaFree(sim->d_pos);
      sim->d_pos = nullptr;
  }
  if ( sim->d_vel ) {
      cudaFree(sim->d_vel);
      sim->d_vel = nullptr;
  }
  if ( sim->d_accel ) {
      cudaFree(sim->d_accel);
      sim->d_accel = nullptr;
  }

  destroy_params<SimulationParams>(sim->d_sim_params);
  sim->d_sim_params = nullptr;
  destroy_params<EnvironmentParams>(sim->d_env_params);
  sim->d_env_params = nullptr;

  return cudaSuccess;
}

cudaError_t 
step(Simulation* sim, uint64_t epoch, const comms::Parameters& com_params) 
{
  cudaError_t status; 

  if ( !sim ) {
    return cudaErrorInvalidValue; 
  }

  if ( (status = sim->grid.build(sim->d_pos, sim->agent_count)) != cudaSuccess ) {
    return status; 
  }

  if ( (status = sim->inbox.clear(0)) != cudaSuccess ) {
    return status; 
  }

  comms::Parameters params = com_params;
    
  params.epoch = epoch; 
  if ( (status = comms::launch(sim->d_pos, sim->d_vel, sim->agent_count, sim->grid,
                               sim->inbox, params, nullptr)) != cudaSuccess ) {
    return status; 
  } 

  if ( (status = boids::launch_compute(
          sim->d_pos, sim->d_vel, sim->inbox,
          sim->d_sim_params, sim->d_env_params,
          sim->d_accel, sim->agent_count, epoch)) != cudaSuccess ) {
    return status; 
  }

  if ( (status = boids::launch_integrate(
          sim->d_pos, sim->d_vel, sim->d_accel,
          sim->agent_count, sim->d_sim_params)) != cudaSuccess ) {
    return status; 
  }

  if ( (status = sim->logger.snapshot(
          sim->d_pos, sim->d_vel, static_cast<uint32_t>(epoch), 
          sim->logger.transfer_stream)) != cudaSuccess ) {
    return status; 
  }

  return cudaSuccess;
}

} // end namespace sim 


namespace {

static inline float3 
get_max_bound(const float3* h_pos, int N)
{
  float3 max = { -FLT_MAX, -FLT_MAX, -FLT_MAX }; 
  
  if ( h_pos && N > 0 ) {
    for (int i = 0; i < N; i++) {
      max.x = std::max(max.x, std::fabs(h_pos[i].x)); 
      max.y = std::max(max.y, std::fabs(h_pos[i].y)); 
      max.z = std::max(max.z, std::fabs(h_pos[i].z)); 
    }
    max *= 2.0; 
  }
  return max; 
}

static inline float3& 
operator*=(float3& a, const float& c)
{
  a.x *= c; 
  a.y *= c; 
  a.z *= c; 
  return a; 
}

}
