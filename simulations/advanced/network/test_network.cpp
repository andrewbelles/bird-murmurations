/*
 * test_network.cpp  Andrew Belles  Sept 28th, 2025 
 *
 * Test of distributed simulated network using spatial grid to 
 * restrict agents to proximity communication only 
 *
 */ 

#include "grid.cuh"
#include "mailbox.cuh"
#include "communication.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <vector> 
#include <cassert> 
#include <iostream> 

void cuda_check(cudaError_t status, const char* loc);

int main(void)
{
  grid::SpatialGrid grid{};
  const float3 bound{4.0, 4.0, 4.0}; 
  const float cell = 1.0; 
  const int N = 64; 
  int idx = 0; 
  std::vector<float3> positions(N); 
  std::vector<float3> velocities(N); 
  cudaError_t status; 

  /********** grid tests **********************************/

  for (int z = 0; z < 4; z++) {
    for (int y = 0; y < 4; y++) {
      for (int x = 0; x < 4; x++) {
        velocities[idx]  = make_float3(x / 5.0, y / 5.0, z / 5.0);
        positions[idx++] = make_float3(x + 0.25, y + 0.25, z + 0.25);
      }
    }
  }

  float3 *d_pos = nullptr, *d_vel = nullptr; 
  
  cuda_check(cudaMalloc(&d_pos, N * sizeof(float3)), 
             "line:40 position malloc");
  cuda_check(cudaMemcpy(d_pos, positions.data(), N * sizeof(float3), 
             cudaMemcpyHostToDevice), "line:42 position memcpy");
  cuda_check(cudaMalloc(&d_vel, N * sizeof(float3)), 
             "line:47 velocity malloc");
  cuda_check(cudaMemcpy(d_vel, velocities.data(), N * sizeof(float3), 
             cudaMemcpyHostToDevice), "line:49 position memcpy");

  // test grid instantiation 
  cuda_check(grid.create(bound, cell, N), "line:55 grid.create");
  cuda_check(grid.build(d_pos, N), "line:57 grid.build");
  std::cout << "[TEST PASSED] Grid created and built\n"; 
  cudaDeviceSynchronize();

  /********** ring buffer tests ***************************/

  network::Mailbox box{};
  const int K = 16;       // capacity per buffer  
  
  cuda_check(box.create(N, K), "line:66 mailbox.create");
  cuda_check(box.clear(0), "line:67 mailbox.clear");
  cudaDeviceSynchronize();

  std::cout << "[TEST PASSED] Mailbox created and cleared\n";

  comms::Parameters params{};
  params.radio_radius = 1.1; 
  params.budget       = 16; 
  params.policy       = network::Overflow::Overwrite; 
  params.epoch        = 1; 
  params.enable_loss  = false; 
  params.loss         = 0.0; 

  cuda_check(comms::launch(d_pos, d_vel, N, grid, box, params, nullptr, 0),
             "line:79 launch communication network");
  cudaDeviceSynchronize();
  std::cout << "[TEST PASSED] Communication network launched\n";

  std::vector<int> head(N), tail(N);
  cuda_check(cudaMemcpy(head.data(), box.head, N * sizeof(int), 
             cudaMemcpyDeviceToHost), "line:85 copy head");
  cuda_check(cudaMemcpy(tail.data(), box.tail, N * sizeof(int), 
             cudaMemcpyDeviceToHost), "line:87 copy tail");

  // lambda to ensure that each agent received exactly the expected number of messages
  auto expected_neighbors = [](int agent_id) {
    int x = agent_id % 4; 
    int y = ( agent_id / 4 ) % 4; 
    int z = ( agent_id / 16 );
    int count = 0; 

    // only inc if count is in correct range 
    count += ( x > 0 ) + ( x < 3 ); 
    count += ( y > 0 ) + ( y < 3 );
    count += ( z > 0 ) + ( z < 3 ); 
    return count; 
  };

  int total_expected = 0, total_recieved = 0, expected = 0; 
  for (int i = 0; i < N; i++) {
    expected = expected_neighbors(i); 
    total_expected += expected; 
    total_recieved += tail[i] - head[i];

    if ( head[i] != 0 ) {
      std::cerr << "Expected head[" << i << "]=0, recieved " << head[i] << '\n';
      exit( 99 );
    }

    if ( tail[i] != expected ) {
      std::cerr << "Agent " << i << "expected " << expected << "messages, recieved" 
                << tail[i] << '\n';
      exit( 99 );
    }
  }

  std::cout << "[TEST PASSED] Index counts match expected counts\n";

  // Explicitly check packet integrity for agent (0) at x=y=z=0 
  {
    int i = 0;
    std::vector<network::Packet> packets(tail[i] - head[i]);
    int base = i * K; 
    cuda_check(cudaMemcpy(packets.data(), box.data + base, 
                          packets.size() * sizeof(network::Packet), 
                          cudaMemcpyDeviceToHost), "line:129 agent0 inbox copy");
    // bitmap for the three positions it should see 
    bool seen[3] = { false, false, false }; 

    for (auto& packet : packets ) {
      if ( packet.epoch != params.epoch ) {
        std::cerr << "Unexpected epoch in agent0 packet: " << packet.epoch << '\n'; 
        exit( 99 ); 
      } 

      switch ( packet.agent_id ) {
        case 1: 
          seen[0] = true; 
          break; 
        case 4: 
          seen[1] = true; 
          break; 
        case 16: 
          seen[2] = true; 
          break; 
        default: 
          std::cerr << "agent0 recieved a packet from unexpected sender " 
                    << packet.agent_id << '\n'; 
          exit( 99 );
      }
    } 

    if ( !(seen[0] && seen[1] && seen[2]) ) {
      std::cerr << "agent0 missing packets from all expected neighbors (id={1,4,16}\n";
      exit( 99 ); 
    }
    std::cout << "[TEST PASSED] Explicit packet check for agent0\n";
  }

  // exit properly 
  grid.destroy(); 
  box.destroy();
  cudaFree(d_pos); 
  cudaFree(d_vel);

  std::cout << "ALL TESTED PASSED\n";
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
