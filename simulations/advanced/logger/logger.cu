/*
 * logger.cu  Andrew Belles  Sept 29th, 2025 
 *
 * Implements the logger interface and the writer CPU thread handler 
 * 
 *
 */

#include "logger.cuh"

#include <atomic>
#include <cuda_runtime.h>
#include <cstdio> 
#include <iostream> 
#include <thread> 
#include <mutex>
#include <condition_variable> 
#include <queue> 
#include <chrono> 
#include <clocale> 

namespace logio {

struct Item { 
  Logger* logger;
  int slot; 
  uint32_t epoch_start; 
};

struct Writer {
  std::FILE* fp = nullptr; 
  int N = 0; 
  int batch_size = 0; 
  size_t bytes_per_batch = 0; 

  size_t rows_per_batch = 0; 
  bool header_written = false; 

  std::mutex m; 
  std::condition_variable cv; 
  std::queue<Item> q; 
  bool stop = false; 

  std::thread t; 
};

static inline cudaError_t nonblocking_stream(cudaStream_t& stream);
static void CUDART_CB on_ready(void* userdata);

cudaError_t 
Logger::create(int N, int batch_size, const char* out_path)
{
  cudaError_t status; 

  this->N = N; 
  this->batch_size = batch_size; 

  rows_per_batch  = static_cast<size_t>(N) * static_cast<size_t>(batch_size);
  bytes_per_batch = rows_per_batch * sizeof(Snapshot); 

  if ( (status = cudaMalloc(&d_buffer, bytes_per_batch)) != cudaSuccess ) {
    return status; 
  }

  if ( (status = cudaMallocHost(&h_buffer[0], bytes_per_batch)) != cudaSuccess) {
    return status; 
  }

  if ( (status = cudaMallocHost(&h_buffer[1], bytes_per_batch)) != cudaSuccess) {
    return status; 
  }

  slots_busy[0].store(0, std::memory_order_relaxed); 
  slots_busy[1].store(0, std::memory_order_relaxed); 

  if ( (status = (nonblocking_stream(transfer_stream))) != cudaSuccess ) {
    return status; 
  } 

  this->w = writer_start(out_path, this->N, this->batch_size);

  return cudaSuccess;
}

void 
Logger::destroy()
{
  if ( transfer_stream ) {
    cudaStreamSynchronize(transfer_stream);
  }

  if ( w ) {
    writer_stop(w);
    w = nullptr;
  }

  if ( transfer_stream ) {
    cudaStreamDestroy(transfer_stream);
    transfer_stream = nullptr;
  }
  
  if ( d_buffer ) {
    cudaFree(d_buffer); 
    d_buffer = nullptr; 
  }

  for (auto& buffer : h_buffer ) {
    if ( buffer ) {
      cudaFreeHost(buffer); 
      buffer = nullptr; 
    }
  }

  slots_busy[0].store(0); 
  slots_busy[1].store(0); 
}

cudaError_t 
Logger::snapshot(const float3* d_pos, const float3* d_vel, 
                 uint32_t epoch, cudaStream_t stream)
{
  // create grid, get modulus index for ring buffer 
  const int block = 256; 
  const int grid = ( N + block - 1 ) / block; 
  const int mod = static_cast<int>(epoch % static_cast<uint32_t>(batch_size));
  cudaError_t status; 

  if ( !d_buffer || !d_pos || !d_vel ) {
    return cudaErrorInvalidDevicePointer; 
  } // avoid device nullptr deref 

  // get the rows that need to be written out from simulation 
  Snapshot* out = d_buffer + static_cast<size_t>(mod) * static_cast<size_t>(N);
  snapshot_rows<<<grid, block, 0, stream>>>(
    out, d_pos, d_vel, N, epoch, mod
  );

  /// TODO Break up into subfunctions 
  if ( mod == batch_size - 1) { // at end of ring index 
    // if int div is odd prefer buffer[1] else [0] 
    int preferred = (( epoch / static_cast<uint32_t>(batch_size)) & 1) ? 1 : 0;
    int slot = -1; 

    // try preferred slot first, then try alternate buffer 
    for (int attempt = 0; attempt < 2; attempt++) {
      int candidate = (attempt == 0) ? preferred : 1 - preferred; 
      int expected  = 0;

      if ( slots_busy[candidate].compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel)) {
        slot = candidate; 
        break; 
      }
    }

    // if both buffers are busy => block until one clears up, implies thread is behind
    while ( slot < 0 ) {
      for (int candidate = 0; candidate < 2; candidate++) {
        int expected = 0; 
        // check for a free buffer 
        if ( slots_busy[candidate].compare_exchange_strong(
              expected, 1, std::memory_order_acq_rel)) {
          slot = candidate; 
          break; 
        }
      } 
      // if didn't find an unblocked buffer, wait 1ms 
      if ( slot < 0 ) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    // copy from device to host buffer 
    status = cudaMemcpyAsync(h_buffer[slot], d_buffer, bytes_per_batch,
                             cudaMemcpyDeviceToHost, transfer_stream);
    if ( status != cudaSuccess ) {
      return status; 
    }

    Item* cb = new Item{ this, slot, epoch - static_cast<uint32_t>(batch_size) };
    
    if ( (status = cudaLaunchHostFunc(transfer_stream, on_ready, cb)) != cudaSuccess) {
      return status; 
    }

    return cudaPeekAtLastError(); 
  }

  return cudaSuccess; 
}

int 
Logger::poll() 
{
  if ( slots_busy[0].load(std::memory_order_relaxed) ) {
    return 0; 
  } else if ( slots_busy[1].load(std::memory_order_relaxed) ) {
    return 1; 
  }
  return -1; 
}

static void 
writer_thread_fn(Writer* w)
{
  std::setlocale(LC_NUMERIC, "C");

  std::string out; 
  out.reserve(w->rows_per_batch * 128);

  while (true) {
    Item item; 
    
    // Ensure queue is non-empty and we haven't asked for a stop 
    {
      std::unique_lock<std::mutex> lock(w->m);
      w->cv.wait(lock, [&]{ return w->stop || !w->q.empty(); });
      if ( w->stop && w->q.empty() ) {
        break; 
      }
      item = w->q.front(); 
      w->q.pop(); 
    }

    Logger* logger = item.logger; 
    const Snapshot* rows = reinterpret_cast<const Snapshot*>(
        logger->h_buffer[item.slot]);

    out.clear();
    out.resize(0); 

    char line[256];
    for (size_t i = 0; i < w->rows_per_batch; i++) {
      const Snapshot& snap = rows[i];
      int n = std::snprintf(
        line, sizeof(line),
        "%u,%u,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
        snap.epoch, 
        snap.agent_id, 
        snap.pos.x, snap.pos.y, snap.pos.z,
        snap.vel.x, snap.vel.y, snap.vel.z 
      );
      out.append(line, static_cast<size_t>(n));
    }

    size_t wrote = std::fwrite(out.data(), 1, out.size(), w->fp);
    if ( wrote != out.size() ) {
      std::cerr << "writer fwrite (csv) short write\n"; 
    }

    logger->slots_busy[item.slot].store(0, std::memory_order_release);
  }
}

Writer* 
writer_start(const char* out_path, int N, int batch_size)
{
  Writer* w = new Writer(); 
  w->fp = std::fopen(out_path, "w");
  if ( !w->fp ) {
    delete w; 
    return nullptr; 
  }

  setvbuf(w->fp, nullptr, _IOFBF, 8 * 1024 * 1024);

  w->N = N; 
  w->batch_size = batch_size; 
  w->rows_per_batch  = static_cast<size_t>(N) * static_cast<size_t>(batch_size);
  w->bytes_per_batch = static_cast<size_t>(N) * static_cast<size_t>(batch_size) *
                       sizeof(Snapshot);

  std::fputs("epoch,agent,x,y,z,vx,vy,vz\n", w->fp);
  w->header_written = true; 

  w->t = std::thread(writer_thread_fn, w);
  return w; 
}

void 
writer_stop(Writer* w)
{
  if ( !w ) {
    return; 
  }

  {
    std::lock_guard<std::mutex> lock(w->m);
    w->stop = true; 
  }

  w->cv.notify_all(); 
  if ( w->t.joinable() ) {
    w->t.join();
  }
  
  if ( w->fp ) { 
    std::fclose(w->fp);
    w->fp = nullptr; 
  }

  delete w; 
}

void 
enqueue_batch(Logger* logger, int slot_idx, uint32_t epoch_start)
{
  // safely enqueue data to be written out 
  {
    std::lock_guard<std::mutex> lock(logger->w->m);
    logger->w->q.push(Item{ logger, slot_idx, epoch_start });
  }

  logger->w->cv.notify_all(); 
}

__global__ void 
snapshot_rows(Snapshot* out_rows, const float3* __restrict__ d_pos, 
              const float3* __restrict__ d_vel, int N, uint32_t epoch, int mod)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  if ( idx >= N ) {
    return; 
  }

  Snapshot snap; 
  snap.epoch    = epoch; 
  snap.agent_id = static_cast<uint32_t>(idx); 
  snap.pos      = d_pos[idx]; 
  snap.vel      = d_vel[idx];
  out_rows[idx] = snap; 
}

static inline cudaError_t 
nonblocking_stream(cudaStream_t& stream) 
{
  return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); 
}

static void CUDART_CB
on_ready(void* userdata)
{
  Item* cbp = static_cast<Item*>(userdata); 
  enqueue_batch(cbp->logger, cbp->slot, cbp->epoch_start);
  delete cbp; 
}

} // end namespace logio 
