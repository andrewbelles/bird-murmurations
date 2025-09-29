/*
 *
 * grid.cu  Andrew Belles  Sept 28th, 2025 
 *
 * Implementation of interface for the SpatialGrid module
 *
 */


#include "grid.cuh"
#include <cuda_runtime.h> 
#include <cmath> 
#include <thrust/device_ptr.h> 
#include <thrust/execution_policy.h> 
#include <thrust/scan.h> 

namespace grid {

/************ host functions ******************************/ 

/*
 * Instantiation of SpatialGrid struct. 
 *
 * Caller Provides: 
 *   maximum simulation bound, discrete cell size, and total number of agents 
 */ 
cudaError_t 
SpatialGrid::create(const float3 max_bound, const float cell_size, const int capacity)
{
  cudaError_t status; 

  this->max_bound = max_bound; 
  this->cell_size = cell_size; 
  
  dims.x = floor( (double)max_bound.x / cell_size );  
  dims.y = floor( (double)max_bound.y / cell_size );  
  dims.z = floor( (double)max_bound.z / cell_size );  
  this->capacity = capacity;
  cell_count = dims.x * dims.y * dims.z;  
  
  // Instantiation on GPU memory of grid pointers 

  if ((status = cudaMalloc(&agent_cell, sizeof(int) * capacity)) != cudaSuccess) {
    destroy();
    return status; 
  }

  if ((status = cudaMalloc(&counts, sizeof(int) * cell_count)) != cudaSuccess) {
    destroy();
    return status; 
  }
  
  if ((status = cudaMalloc(&offsets, sizeof(int) * (cell_count + 1))) != cudaSuccess) {
    destroy();
    return status; 
  }

  if ((status = cudaMalloc(&agents, sizeof(int) * capacity)) != cudaSuccess) {
    destroy();
    return status; 
  }

  if ((status = cudaMalloc(&cell_write_ptr, sizeof(int) * cell_count)) != cudaSuccess) {
    destroy();
    return status; 
  }

  return cudaSuccess; 
}

/*
 * Rebuilds the spatial grid at a single timestep  
 *
 * Caller provides: 
 *   Agent positions (as a device pointer) and the number of agents 
 *   Optionally, a different cudaStream (defaults to 0)
 */
cudaError_t 
SpatialGrid::build(const float3* positions, int N, cudaStream_t stream)
{
  const int block = 256; 
  int grid = 0; 
  cudaError_t status; 

  // guard against invalid instantiation 
  if ( N > capacity ) {
    return cudaErrorInvalidValue; 
  }

  // zero out grid 
  {
    grid = (cell_count + block - 1) / block; 
    reset_counts<<<grid, block, 0, stream>>>(counts, cell_count); 
  }

  // assign cells and count total in each 
  {
    grid = (N + block - 1) / block; 
    assign_cells<<<grid, block, 0, stream>>>(
      positions, N, cell_size, dims, 
      agent_cell, counts
    );
  }

  // calculate prefix sum for offsets 
  {
    thrust::device_ptr<int> d_counts(counts);
    thrust::device_ptr<int> d_offsets(offsets); 
    thrust::exclusive_scan(
      thrust::device.on(stream),
      d_counts, d_counts + cell_count, d_offsets
    );
    if ( (status = cudaMemcpyAsync(
            offsets + cell_count, &N, sizeof(int),
            cudaMemcpyHostToDevice, stream)) != cudaSuccess ) {
      destroy();
      return status; 
    }
  }

  // copy offsets to cell_write_ptr 
  {
    if ( (status = cudaMemcpyAsync(
            cell_write_ptr, offsets, sizeof(int) * cell_count,
            cudaMemcpyDeviceToDevice, stream)) != cudaSuccess ) {
      destroy();
      return status; 
    }
  }

  // fill agent lists per cell using computed offsets 
  {
    grid = (N + block - 1) / block; 
    fill_cells<<<grid, block, 0, stream>>>(
      agent_cell, N, offsets, cell_write_ptr, agents
    ); 
  }
  
  return cudaPeekAtLastError();
}

/*
 * Frees GPU memory allocated at instantiation 
 */
void 
SpatialGrid::destroy()
{
  cudaFree(agent_cell);
  cudaFree(offsets);
  cudaFree(counts);
  cudaFree(agents);
  cudaFree(cell_write_ptr);
}

/************ device functions ****************************/ 

__device__ inline int3 
get_cell(const float3 position, const float cell_size)
{
  int3 cell; 
  cell.x = __float2int_rd(position.x / cell_size);
  cell.y = __float2int_rd(position.y / cell_size);
  cell.z = __float2int_rd(position.z / cell_size);
  return cell; 
}


__device__ inline int 
cell_to_index(const int3 cell, const int3 dims)
{
  return cell.x + dims.x * (cell.y + dims.y * cell.z); 
}


__device__ inline bool 
inBounds(const int3 cell, const int3 dims)
{
  return (unsigned)cell.x < (unsigned)dims.x && 
         (unsigned)cell.y < (unsigned)dims.y && 
         (unsigned)cell.z < (unsigned)dims.z; 
}


/************ kernels *************************************/ 

__global__ void 
reset_counts(int* __restrict__ counts, int cell_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( idx < cell_count ) {
    counts[idx] = 0; 
  }
}


__global__ void 
assign_cells(const float3* __restrict__ positions, int N, float cell_size, int3 dims, 
             int* __restrict__ agent_cell, int* __restrict__ counts)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  float3 pos; 
  int3 coord; 
  int cell; 

  if ( idx >= N ) {
    return; 
  } // guard nullptr deref 

  // get cell coordinate 
  pos = positions[idx]; 
  coord = get_cell(pos, cell_size);

  // clamp to avoid deref  
  coord.x = (coord.x < 0) ? 0 : ( coord.x >= dims.x ? dims.x - 1 : coord.x );
  coord.y = (coord.y < 0) ? 0 : ( coord.y >= dims.y ? dims.y - 1 : coord.y );
  coord.z = (coord.z < 0) ? 0 : ( coord.z >= dims.z ? dims.z - 1 : coord.z );

  // inc count for this cell 
  cell = cell_to_index(coord, dims);
  atomicAdd(&counts[cell], 1);
}

__global__ void 
fill_cells(const int* __restrict__ agent_cell, int N, const int* __restrict__ offsets, 
           int* __restrict__ cell_write_ptr, int* __restrict__ agents) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  int cell;
  int slot; 

  if ( idx >= N ) {
    return; 
  } // guard nullptr deref 
  
  cell = agent_cell[idx]; 
  slot = atomicAdd(&cell_write_ptr[cell], 1);
  agents[slot] = idx; 
}

} 
