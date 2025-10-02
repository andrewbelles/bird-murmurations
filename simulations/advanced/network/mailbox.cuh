/*
 * mailbox.cuh  Andrew Belles  Sept 28th, 2025 
 *
 * Module to define per agent ring buffers to simulate 
 * intake of packets from bluetooth 
 *
 */

#ifndef __MAILBOX_CUH 
#define __MAILBOX_CUH 

#include <cuda_runtime.h>
#include <cstdint> 

namespace network {

/************ structure definitions ***********************/

/*
 * Formal Packet definition 
 */
struct Packet {
 int agent_id;    // id of sender  
 float3 x, v;     // position and velocity 
 uint32_t epoch;  // sequence in time  
}; 

/*
 * Enum helper for clarity when determining handling packets in full buffer 
 */
enum class Overflow : uint8_t {
  Overwrite = 0, 
  Drop      = 1 
};

/*
 * Ring buffer per agent in space 
 *
 */ 
struct Mailbox {
  Packet* data; // [ agents * box size]
  int* head;      
  int* tail; 
  int capacity; // capacity per agent 
  int size;     // number of agents 

  cudaError_t create(int N, int K); 
  void destroy(); 

  // sensible call to kernel helper  
  cudaError_t clear(cudaStream_t stream=0);
}; 

/************ device functions ****************************/

__device__ __forceinline__ int 
inbox_base(const Mailbox& box, int agent_id)
{
  return agent_id * box.capacity; 
}

__device__ __forceinline__ int 
ring_index(int base, int index, int K)
{
  return base + (index % K);
}

__device__ bool push(Mailbox& box, int recipient_id, 
                     const Packet& packet, Overflow policy);

// Start draining 
__device__ void drain(const Mailbox& box, int agent_id, int& head_out, int& tail_out);

// Advance consumer head index by n values 
__device__ void advance(Mailbox& box, int agent_id, int n); 

// iterator over packets using user provided functor 
// functor: void(int index, const Packet& packet) 
template <class F> __device__ int 
drain_iterator(const Mailbox& box, int agent_id, int n_packets, F&& fn)
{
  int head = 0, tail = 0, available = 0, slot = 0; 
  
  drain(box, agent_id, head, tail);
  available = tail - head;
  if ( available <= 0 ) {
    return 0; 
  }

  // determine the number of available packets to iterate over
  const int take = (n_packets < available ? n_packets : available); 
  const int base = inbox_base(box, agent_id);

  // iterate over ring buffer 
  for (int i = 0; i < take; i++) {
    slot = ring_index(base, head + i, box.capacity);
    const Packet& packet = box.data[slot]; 
    fn(slot, packet);   // call to iterator helper  
  }

  // advance ring buffer to consume the iterated packets 
  advance(const_cast<Mailbox&>(box), agent_id, take);
  return take;  // return number of packets read  
}

/************ kernel functions ****************************/

// kernel helper to clear mailbox quickly 
__global__ void clear_kernel(int* head, int* tail, int N); 

} // namespace network 

#endif // __MAILBOX_CUH
