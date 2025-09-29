/*
 * test_logger.cpp  Andrew Belles  Sept 29th, 2025 
 *
 * Streams deterministic data into logger to test complex threaded io operations 
 *
 *
 */ 

#include "logger.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <vector> 
#include <cassert> 
#include <iostream> 

void cuda_check(cudaError_t status, const char* loc);

int main(int argc, char* argv[]) 
{
  const int N = 2048; 
  const int B = 32; 

  cudaStream_t sim = nullptr; 
  cuda_check(cudaStreamCreateWithFlags(&sim, cudaStreamNonBlocking), 
             "line:24 sim stream creation");

  logio::Logger logger{};
  float3 *d_pos = nullptr, *d_vel = nullptr; 
  cuda_check(cudaMalloc(&d_pos, N * sizeof(float3)), "line:30 d_pos malloc"); 
  cuda_check(cudaMalloc(&d_vel, N * sizeof(float3)), "line:31 d_vel malloc"); 
  cuda_check(logger.create(N, B, "../data/boids.bin"), "line:32 logger.create");
  
  std::cout << "[TEST PASSED] Instantiation of Logger\n";
}

void 
cuda_check(cudaError_t status, const char* loc)
{
  if ( status != cudaSuccess ) {
    std::cerr << "CUDA error @ " << loc << ": " << cudaGetErrorString(status);
    exit( 99 );
  }
}
