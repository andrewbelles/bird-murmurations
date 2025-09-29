/*
 * test_logger.cpp  Andrew Belles  Sept 29th, 2025 
 *
 * Streams deterministic data into logger to test complex threaded io operations 
 *
 *
 */ 

#include "logger.cuh"

#include <cuda_runtime.h>
#include <vector> 
#include <cassert> 
#include <iostream> 

void cuda_check(cudaError_t status, const char* loc);

namespace unit {
__global__ void fill_states(float3* pos, float3* vel, int N, uint32_t epoch);
}

int main(int argc, char* argv[]) 
{
  const int N = 2048; 
  const int B = 32; 
  const int EPOCHS = 64; 
  const int block = 128, grid = (N + block - 1) / block; 
 
  if ( argc != 2 ) {
    std::cerr << "invalid usage: ./logger_test output.bin\n";
    exit( 99 );
  }

  const char* path = argv[1];

  cudaStream_t sim = nullptr; 
  cuda_check(cudaStreamCreateWithFlags(&sim, cudaStreamNonBlocking), 
             "line:24 sim stream creation");

  logio::Logger logger{};
  float3 *d_pos = nullptr, *d_vel = nullptr; 
  cuda_check(cudaMalloc(&d_pos, N * sizeof(float3)), "line:30 d_pos malloc"); 
  cuda_check(cudaMalloc(&d_vel, N * sizeof(float3)), "line:31 d_vel malloc"); 
  cuda_check(logger.create(N, B, path), "line:32 logger.create");

  std::cout << "[TEST PASSED] Instantiation of Logger\n";

  for (uint32_t e = 0; e < uint32_t(EPOCHS); e++) {
    unit::fill_states<<<grid, block, 0, sim>>>(d_pos, d_vel, N, e); 
    cuda_check(cudaPeekAtLastError(), "line:44 fill states");
    cuda_check(logger.snapshot(d_pos, d_vel, e, sim), "line:45 logger.snapshot");
  }

  cuda_check(cudaStreamSynchronize(sim), "line:51 sync sim");
  cuda_check(cudaStreamSynchronize(logger.transfer_stream), "line:52 sync transfer");

  std::cout << "[TEST PASSED] Successfully ran logging pipeline\n";

  logger.destroy();

  FILE* fptr = std::fopen(path, "rb");
  if ( !fptr ) {
    std::cerr << "[TEST FAILED] failed to open file\n"; 
    exit( 1 ); 
  }

  std::fseek(fptr, 0, SEEK_END);
  long file_size = std::ftell(fptr); 
  std::fseek(fptr, 0, SEEK_SET);

  const size_t rows_per_batch = size_t(N) * size_t(B);
  const size_t expected_bytes = rows_per_batch * sizeof(logio::Snapshot) * (EPOCHS / B); 
  if ( (size_t)file_size != expected_bytes ) {
    std::cerr << "[TEST FAILED] Expected: " << expected_bytes 
              << " but got " << file_size << " bytes\n";
    exit( 2 );
  }

  std::cout << "[TEST PASSED] File size matches expected size\n"; 

  std::vector<logio::Snapshot> rows(rows_per_batch * (EPOCHS / B)); 
  size_t read_bytes = std::fread(
    rows.data(), 1, rows.size() * sizeof(logio::Snapshot), fptr
  );
  std::fclose(fptr); 

  if ( read_bytes != rows.size() * sizeof(logio::Snapshot) ) {
    std::cerr << "[TEST FAILED] read bytes is short of expected\n";
    exit( 3 );
  }

  std::cerr << "[TEST PASSED] Bytes read matches expected\n"; 

  auto index = [&](uint32_t batch, uint32_t mod, uint32_t agent_id) -> size_t {
    return batch * rows_per_batch + mod * uint32_t(N) + agent_id; 
  }; 

  {
    // epoch 0, agent 0
    const auto& snap = rows[index(0, 0, 0)];
    assert(snap.epoch    == 0);
    assert(snap.agent_id == 0u);
    assert(snap.pos.x == 0.0 && snap.pos.y == 0.0 && snap.pos.z == 1.0);
    assert(snap.vel.x == 0.0 && snap.vel.y == 0.0 && snap.vel.z == -1.0);
  }

  {
    // epoch 31, agent 2047 (last agent)
    const auto& snap = rows[index(0, B-1, N-1)];
    assert(snap.epoch    == 31);
    assert(snap.agent_id == uint32_t(N-1));
    assert(snap.pos.x == 31.0 && snap.pos.y == float(N-1) && snap.pos.z == 1.0);
    assert(snap.vel.x == float(N-1) && snap.vel.y == 31.0 && snap.vel.z == -1.0);
  }

  {
    // epoch 32, agent 123
    const uint32_t a = 123;
    const auto& snap = rows[index(1, 0, a)];
    assert(snap.epoch    == 32);
    assert(snap.agent_id == a);
    assert(snap.pos.x == 32.0 && snap.pos.y == float(a) && snap.pos.z == 1.0);
    assert(snap.vel.x == float(a) && snap.vel.y == 32.0 && snap.vel.z == -1.0);
  }

  {
    // epoch 63, agent 77
    const uint32_t a = 77;
    const auto& snap = rows[index(1, B-1, a)];
    assert(snap.epoch    == 63);
    assert(snap.agent_id == a);
    assert(snap.pos.x == 63.0 && snap.pos.y == float(a) && snap.pos.z == 1.0);
    assert(snap.vel.x == float(a) && snap.vel.y == 63.0 && snap.vel.z == -1.0);
  }
  std::cout << "[TEST PASSED] Spot check of written packets against expected\n";

  std::cout << "ALL TESTS PASSED\n";

  cudaFree(d_pos);
  cudaFree(d_vel);
  cudaStreamDestroy(sim);

  return 0; 
}

void 
cuda_check(cudaError_t status, const char* loc)
{
  if ( status != cudaSuccess ) {
    std::cerr << "CUDA error @ " << loc << ": " << cudaGetErrorString(status);
    exit( 99 );
  }
}

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
