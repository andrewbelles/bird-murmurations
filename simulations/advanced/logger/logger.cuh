/*
 * logger.cuh  Andrew Belles  Sept 29th, 2025 
 *
 * Defines the GPU logger which streams data from simulation back to CPU 
 * to be written to binary 
 *
 */ 

#ifndef __LOGGER_CUH 
#define __LOGGER_CUH 

#include <cuda_runtime.h> 
#include <cstdint> 
#include <atomic> 

namespace logio {

struct Writer;

Writer* writer_start(const char* out_path, int N, int batch_start);
void writer_stop(Writer* w); 

struct Snapshot {
  uint32_t epoch; 
  uint32_t agent_id; 
  float3 pos; 
  float3 vel; 
};

struct Logger {
  int N = 0; 
  int batch_size = 0; 
  size_t rows_per_batch = 0; 
  size_t bytes_per_batch = 0; 
  Writer* w;

  // device buffer and host ring buffer for incoming data 
  Snapshot* d_buffer    = nullptr; 
  Snapshot* h_buffer[2] = { nullptr, nullptr }; 
  std::atomic<int> slots_busy[2] = { 0, 0 };
  cudaStream_t transfer_stream = nullptr;

  cudaError_t create(int N, int batch_size, const char* out_path);
  void destroy(); 
  
  cudaError_t snapshot(const float3* d_pos, const float3* d_vel, 
                       uint32_t epoch, cudaStream_t stream);
  
  // get slot index for last enqueued copy 
  int poll();  
};

__global__ void snapshot_rows(Snapshot* out_rows, const float3* __restrict__ d_pos, 
                              const float3* __restrict__ d_vel, int N, uint32_t epoch,
                              int mod);

void enqueue_batch(Logger* logger, int slot_idx, uint32_t epoch_start);

} // end namespace logio 

namespace unit {

__global__ void 
fill_states(float3* pos, float3* vel, int N, uint32_t epoch)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( idx >= N ) {
    return; 
  }

  pos[idx] = make_float3(float(epoch), float(idx), 1.0);
  vel[idx] = make_float3(float(idx), float(epoch), -1.0);
}

}

#endif // __LOGGER_CUH
