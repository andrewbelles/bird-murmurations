#ifndef __GRID_CUH 
#define __GRID_CUH 

/*
 *
 * grid.cuh  Andrew Belles  Sept 28th, 2025 
 *
 * Defines the interface for spatial grid module that 
 * restricts communication range of agents in space 
 */ 

#include <cuda_runtime.h> 

namespace grid {

/*
 * SpatialGrid interface and host metadata. 
 *
 * We assume that the lower bound of the grid is the origin 
 */ 
struct SpatialGrid {
  /********** simulation parameters ***********************/ 
  float3 max_bound; 
  float cell_size; 
  int3 dims;                // floor(max_bound / cell_size) 
  int cell_count, capacity; // _, 2048 

  /********** device ptrs *********************************/
  int *agent_cell, *counts, *offsets, *agents, *cell_write_ptr; 

  // cpu call for gpu constructor and destructor 
  cudaError_t create(const float3 max_bound, const float cell_size, const int capacity);
  void destroy(); 
  
  cudaError_t build(const float3* positions, int N, cudaStream_t stream=0);

  /********** host functions ******************************/ 
  // [start, end) range 
  __host__ inline void get_cell_range(int cell_index, int& start, int& end) const;
  
  // get cell index from 3d position 
  __host__ inline int position_to_cell(const float3& position) const; 

};

/********** device functions ****************************/ 
// get cell id from position and cell resolution  
__device__ inline int3 get_cell(const float3 position, const float cell_size);

// Generate an integer cell integer using cell id and total dims 
__device__ inline int cell_to_index(const int3 cell, const int3 dims); 

// Simple check for an in bounds value 
__device__ inline bool inBounds(const int3 cell, const int3 dims); 

/********** kernels *************************************/
// clears grid 
__global__ void reset_counts(int* __restrict__ counts, int cell_count);

// computes each agent's cell id and increments counts (atomically)
__global__ void assign_cells(const float3* __restrict__ positions, int N,
                             float cell_size, int3 dims, 
                             int* __restrict__ agent_cell, 
                             int* __restrict__ counts);

// fill contiguous agent lists per cell
__global__ void fill_cells(const int* __restrict__ agent_cell, int N, 
                           const int* __restrict__ offsets, 
                           int* __restrict__ cell_write_ptr,
                           int* __restrict__ agents);
} // end namespace grid 

#endif 
