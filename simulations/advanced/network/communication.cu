/*
 * communication.cu  Andrew Belles  Sept 29th, 2025 
 *
 * Defines the implementation of methods from communication.cuh  
 * which aim to unify simulation of communication network for agents 
 *
 */

#include "communication.cuh"
#include <cuda_runtime.h>

namespace {

__host__ __device__ static __forceinline__ float3& 
operator+=(float3& a, const float3& b)
{
  a.x += b.x; 
  a.y += b.y; 
  a.z += b.z; 
  return a; 
}

}

namespace comms {

/*
 * Sensible call to kernel to launch network 
 *
 * Caller Provides: 
 *   All values required by broadcast kernel (See definition)
 * 
 * We return: 
 *   cudaStatus from kernel call   
 */
cudaError_t
launch(const float3* positions, const float3* velocities, int N, 
       const grid::SpatialGrid& grid, network::Mailbox& inboxes, 
       const Parameters& params, uint32_t* rng, cudaStream_t stream)
{
  const int block = 256; 
  const int grid_size = ( N + block - 1) / block; 

  if ( !positions || !velocities || N <= 0 ) {
    return cudaErrorInvalidValue;  
  } // avoid nullptr deref of agent information 
  
  if ( !grid.offsets || !grid.agents ) {
    return cudaErrorInvalidDevicePointer;
  } // avoid nullptr deref of grid information 
  
  // call to kernel 
  broadcast_positions<<<grid_size, block, 0, stream>>>(
    positions, velocities, N, grid.cell_size, grid.dims, grid.offsets, 
    grid.agents, inboxes, params, rng 
  );

  return cudaPeekAtLastError();
}

static inline __device__ float 
update_rng(uint32_t* state)
{
  (*state) = (*state) * 1664525 + 1013904223; 
  return static_cast<float>((*state) & 0x00FFFFFF) / float(0x01000000);
}
/*
 * GPU KERNEL
 * 
 * Distributed over all agents, tries to communicate position and velocity with 
 * surrounding agents as informed by spatial grid. 
 *
 * Caller Provides: 
 *   agent positions and velocities, agent count, cell resolution and dimension of space
 *   prefix sums (offsets), agent cell index (agents), ring buffers (inboxes), and 
 *   network parameters 
 *
 * We modify: 
 *   For each agent, we attempt to provide their position and velocity to the ring 
 *   buffer of all other agents in range. 
 *
 * If params.enable_loss = true:
 *   Enables a lossy and noisy channel to simulate loss of infomration in field. 
 */
__global__ void 
broadcast_positions(const float3* __restrict__ positions, 
                    const float3* __restrict__ velocities, int N, float cell_size, 
                    int3 dims, const int* __restrict__ offsets, 
                    const int* __restrict__ agents, 
                    network::Mailbox inboxes, Parameters params, uint32_t* rng_state)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( idx >= N ) {
    return; 
  }

  const float3 xi = positions[idx]; 
  
  // lambda communication function to iterate with  
  auto communicate = [&] __device__ (int neighbor) -> bool {
    const float3 xj = positions[neighbor];
    const float dx = xi.x - xj.x; 
    const float dy = xi.y - xj.y; 
    const float dz = xi.z - xj.z; 
    const float l2 = dx * dx + dy * dy + dz * dz; 
    const float r2 = params.radio_radius * params.radio_radius; 
    if ( l2 > r2 ) {
      return false; 
    }

    if ( params.enable_loss && update_rng(&rng_state[idx]) < params.loss ) {
      return false;  // skip this packet  
    }

    // push packet to neighbor 
    network::Packet packet; 
    packet.agent_id = idx; 
    packet.x = xi; 
    packet.v = velocities[idx];
    packet.epoch = params.epoch;
    
    if ( params.enable_noise && params.noise > 0.0 ) {
      float3 noise_inject = { 
        params.noise * update_rng(&rng_state[idx]), 
        params.noise * update_rng(&rng_state[idx]), 
        params.noise * update_rng(&rng_state[idx]) 
      }; 

      packet.x += noise_inject;
    }

    return network::push(inboxes, neighbor, packet, params.policy); 
  };

  // iterator over 3x3x3 grid 
  neighbor_agent_iterator<1>(
    idx, xi, params.radio_radius, cell_size, dims, 
    offsets, agents, communicate, params.budget
  );
}

}
