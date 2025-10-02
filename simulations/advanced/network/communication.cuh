/*
 * communication.cuh  Andrew Belles  Sept 28th, 2025 
 *
 * Defines methods for communicating with agents in proximity 
 *
 *
 */ 

#ifndef __COMMUNICATION_CUH 
#define __COMMUNICATION_CUH

#include <cuda_runtime.h> 
#include <cstdint> 
#include "grid.cuh"
#include "mailbox.cuh"

namespace comms {

struct Parameters {
  float radio_radius; 
  int budget; 
  network::Overflow policy; 
  uint32_t epoch; 

  /********** lossy/noisy channel *************************/
  bool enable_loss = false;
  float loss = 0.0;

  bool enable_noise = false; 
  float noise = 0.0; 
};

cudaError_t launch(const float3* positions, const float3* velocities, int N, 
                   const grid::SpatialGrid& grid, network::Mailbox& inboxes, 
                   const Parameters& params, uint32_t* rng, cudaStream_t stream=0);

/*
 * Iterator per cell over R (radius) of cells
 * F:= __device__ bool fn(int neighbor); 
 *
 * Suggest unrolls for nested for loops to mitigate deep looping  
 */
template<int R, class F> 
__device__ __forceinline__
void neighbor_cell_iterator(int3 base, int3 dims, F&& fn)
{
  int x = 0, y = 0, z = 0, cell = 0; 

  #pragma unroll 
  for (int dz = -R; dz <= R; dz++) {
    z = base.z + dz; 
    if ( (unsigned)z >= (unsigned)dims.z ) {
      continue; 
    }

    #pragma unroll
    for (int dy = -R; dy <= R; dy++) {
      y = base.y + dy; 
      if ( (unsigned)y >= (unsigned)dims.y ) {
        continue; 
      }

      #pragma unroll 
      for (int dx = -R; dx <= R; dx++) {
        x = base.x + dx; 
        if ( (unsigned)x >= (unsigned)dims.x ) {
          continue; 
        }

        // get true cell index and call iterator helper 
        cell = grid::cell_to_index({x, y, z}, dims); 
        fn(cell);
      }
    }


  }
}

/*
 * Iterator per agent over radius of cells 
 * F:= __device__ bool fn(int neighbor);
 *
 */
template<int R, class F>
__device__ __forceinline__ 
int neighbor_agent_iterator(int self_id, float3 self_pos, float radius, 
                             float cell_size, int3 dims, 
                             const int* __restrict__ offsets, 
                             const int* __restrict__ agents, 
                             F&& fn, int budget)
{
  // ensure existing budget 
  if ( budget <= 0 ) {
    return 0; 
  }

  const int3 base = grid::get_cell(self_pos, cell_size); 
  int consumed = 0; 

  // lambda helper to call iterator 
  auto functor = [&] __device__ (int cell) {
    const int start = offsets[cell]; 
    const int end   = offsets[cell + 1]; 

    for (int i = start; i < end; i++) {
      const int j = agents[i];
      if ( j == self_id ) {
        continue; 
      }

      bool used = fn(j); 

      if ( used ) {
        if ( ++consumed >= budget ) {
          return; 
        }
      }
    }
  };
  
  neighbor_cell_iterator<R>(base, dims, functor); 
  return consumed; 
}

__global__ void broadcast_positions(const float3* __restrict__ positions, 
                                    const float3* __restrict__ velocities, 
                                    int N, float cell_size, int3 dims, 
                                    const int* __restrict__ offsets, 
                                    const int* __restrict__ agents, 
                                    network::Mailbox inboxes, 
                                    Parameters params, uint32_t* rng_state);

}

#endif // __COMMUNICATION_CUH
