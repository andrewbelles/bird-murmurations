/*
 * simulation.cuh 
 * 
 * Defines Interface for initializing, stepping, and ending simulation 
 *
 *
 */ 

#ifndef __SIMULATION_CUH 
#define __SIMULATION_CUH

#include "network/grid.cuh"
#include "network/mailbox.cuh"
#include "network/communication.cuh" 
#include "logger/logger.cuh"
#include "agent/environment.cuh"

#include <cuda_runtime.h>
#include <string> 

namespace sim {

struct Simulation {
  int agent_count, bufr_size; 
  
  float3 *d_pos, *d_vel, *d_accel; 
  
  grid::SpatialGrid grid; 
  network::Mailbox inbox; 

  SimulationParams* d_sim_params; 
  EnvironmentParams* d_env_params;
  logio::Logger logger; 
};

cudaError_t create(Simulation* sim, const float3* h_pos, const float* h_vel, 
                   int agent_count, int bufr_size, const std::string& logger_path,
                   const SimulationParams* sim_params, 
                   const EnvironmentParams* env_params);
cudaError_t destroy(Simulation* sim);

cudaError_t step(Simulation* sim, uint64_t epoch, const comms::Parameters& com_params,
                 uint32_t* rng_state);

} // end namespace sim

#endif // __SIMULATION_CUH
