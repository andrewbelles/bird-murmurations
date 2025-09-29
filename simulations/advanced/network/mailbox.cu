/*
 * mailbox.cu  Andrew Belles  Sept 28th, 2025 
 *
 * Implements Mailbox interface defined in mailbox.cuh 
 *
 *
 */ 

#include "mailbox.cuh"
#include <cuda_runtime.h> 

namespace network {

cudaError_t 
Mailbox::create(int N, int K)
{
  this->size = N; 
  this->capacity = K; 

  cudaError_t status; 
  size_t message_byte_count = sizeof(Packet) * size_t(N) * size_t(K);
  size_t index_byte_count   = sizeof(int) * size_t(N);

  if ( (status = cudaMalloc(&data, message_byte_count)) != cudaSuccess ) {
    destroy();
    return status; 
  }

  if ( (status = cudaMalloc(&head, index_byte_count)) != cudaSuccess ) {
    destroy(); 
    return status; 
  }

  if ( (status = cudaMalloc(&tail, index_byte_count)) != cudaSuccess ) {
    destroy();
    return status; 
  } 

  return clear(0); 
}

void 
Mailbox::destroy()
{
  if ( data ) {
    cudaFree(data); 
    data = nullptr; 
  }
  if ( head ) {
    cudaFree(head); 
    head = nullptr; 
  }
  if ( tail ) {
    cudaFree(tail); 
    tail = nullptr; 
  }
}

cudaError_t 
Mailbox::clear(cudaStream_t stream)
{
  const int block = 256; 
  const int grid = ( size + block - 1 ) / block; 
  clear_kernel<<<grid, block, 0, stream>>>(head, tail, size);
  return cudaPeekAtLastError();
}

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

__device__ bool 
push(Mailbox& box, int recipient_id, const Packet& packet, Overflow policy)
{
  const int K = box.capacity; 
  const int base = inbox_base(box, recipient_id);
  int head = 0, slot = 0; 
  int tail = atomicAdd(&box.tail[recipient_id], 1); 

  if ( policy == Overflow::Drop ) {
    head = box.head[recipient_id];
    if ( tail - head >= K) {
      return false; 
    }
  }

  slot = ring_index(base, tail, K); 
  box.data[slot] = packet; 
  return true;
}

__device__ void 
drain(const Mailbox& box, int agent_id, int& head_out, int& tail_out)
{
  int head = box.head[agent_id], tail = box.tail[agent_id]; 
  const int K = box.capacity; 
  int overflow = ( tail - head ) - K; 
  
  if ( overflow > 0 ) {
    head += overflow; 
  }

  head_out = head; 
  tail_out = tail; 
}

__device__ void 
advance(Mailbox& box, int agent_id, int n)
{
  box.head[agent_id] += n; 
}

__global__ void 
clear_kernel(int* head, int* tail, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( idx < N ) {
    head[idx] = 0; 
    tail[idx] = 0; 
  }
}

} // end namespace network 
