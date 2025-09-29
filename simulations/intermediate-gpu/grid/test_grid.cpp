/*
 * test_grid.cpp  Andrew Belles  Sept 28th, 2025 
 *
 * Very simple unit test for SpatialGrid interface 
 *
 *
 */ 

#include "grid.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector> 
#include <cassert> 
#include <iostream> 

int main(void)
{
  grid::SpatialGrid grid{};
  const float3 bound{2.0, 2.0, 2.0}; 
  const float cell = 1.0; 
  const int N = 8; 
  int idx = 0; 
  std::vector<float3> positions(N); 
  cudaError_t status; 

  for (int z = 0; z < 2; z++) {
    for (int y = 0; y < 2; y++) {
      for (int x = 0; x < 2; x++) {
        positions[idx++] = make_float3(x + 0.25, y + 0.25, z + 0.25);
      }
    }
  }

  float3* device_positions = nullptr; 
  cudaMalloc(&device_positions, N * sizeof(float3));
  cudaMemcpy(device_positions, positions.data(), N * sizeof(float3), 
             cudaMemcpyHostToDevice);

  assert( (status = grid.create(bound, cell, N)) == cudaSuccess );
  std::cout << "[TEST PASSED] Instantation of Grid\n"; 
  assert( (status = grid.build(device_positions, N)) == cudaSuccess );
  std::cout << "[TEST PASSED] Grid Built\n"; 
  cudaDeviceSynchronize();

  grid.destroy(); 
  cudaFree(device_positions); 
  return 0; 
}
