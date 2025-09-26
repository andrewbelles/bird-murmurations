/*
 *
 * boids_simple.cpp  Andrew Belles  Sept 26th, 2025 
 *
 * Simple boids 2D simulation aiming to show independent agents communicating via mesh 
 * to determine alignment, cohesion, and separation benchmarks for linear case 
 */ 

#include <thread> 
#include <iostream> 
#include <vector> 
#include <atomic> 
#include <algorithm>

struct vec3 {
  double x, y, z; 

  friend vec3 operator+(vec3 a, const vec3 b) 
  { 
    return {a.x + b.x, a.y + b.y, a.z + b.z}; 
  }
  friend vec3 operator-(vec3 a, const vec3 b)
  { 
    return {a.x - b.x, a.y - b.y, a.z - b.z}; 
  }

}; 

typedef uint16_t Agent; 
typedef std::tuple<Agent, vec3, vec3> Packet; 

class Pool {
private: 
  size_t capacity_{8}; 
  Packet* bufr_[2];
  std::atomic<size_t> widx_[2]; 
  std::atomic<int> active_; 
  std::atomic<size_t> count_[2];
  std::atomic<int> readers_[2];

public: 

  explicit Pool(size_t capacity=8) 
    : capacity_(capacity), 
      bufr_{new Packet[capacity], new Packet[capacity]}, 
      widx_{0, 0},
      active_(0) {}

  ~Pool() {
    delete[] bufr_[0];
    delete[] bufr_[1];
  }

  bool 
  push(const Packet& packet) noexcept 
  {
    int a = active_.load(std::memory_order_acquire);
    auto idx = widx_[a].fetch_add(1, std::memory_order_acq_rel);

    if ( idx >= capacity_ ) {
      return false; 
    } else {
      bufr_[a][idx] = packet; 
      return true; 
    }
  }

  void
  rotate(int agent_count)
  {
    int prev = active_.load(std::memory_order_relaxed);
    int next = 1 - prev; 

    active_.store(next, std::memory_order_acq_rel);

    size_t n = std::min(
        widx_[prev].load(std::memory_order_acquire),
        capacity_
    );

    count_[prev].store(n, std::memory_order_release); 
    widx_[next].store(0, std::memory_order_release);
  }

  std::vector<Packet> 
  fetch()
  {
    int inactive = 1 - active_.load(std::memory_order_acquire);
    size_t n = count_[inactive].load(std::memory_order_acquire);

    std::vector<Packet> out; 
    out.reserve(n);
    for (size_t i(0); i < n; i++) {
      out.push_back(bufr_[inactive][i]);
    }

    if (readers_[inactive].fetch_sub(1, std::memory_order_acq_rel) == 1) {
      widx_[inactive].store(0, std::memory_order_release);
      count_[inactive].store(0, std::memory_order_release);
    }
    return out; 
  }
};

class SwarmAgent {
private: 
  Agent id_{0};  
  vec3 position_{0.0, 0.0, 0.0};
  vec3 velocity_{0.0, 0.0, 0.0};
  Pool& pool_; 

public: 

  SwarmAgent(Agent id, vec3 x0, Pool& pool) 
    : id_(id),
      position_(x0),
      velocity_({0.0, 0.0, 0.0}),
      pool_(pool) {}

  /* 
   * Listen for positions from other agents
   */ 
  std::vector<Packet>  
  communicate()
  {
    std::vector<Packet> out; 
    Packet packet = std::make_tuple(id_, position_, velocity_);
    pool_.push(packet);

    while ( out.size() != 8 ) {
      out = pool_.fetch();   // fetch until it returns something 
    }     
    return out; 
  }

  void 
  boids()
  {
    


  }

  Agent id() const { return id_; }
  vec3 pos() const { return position_; }
};


int main(void)
{
  std::vector<std::thread> threads(8);  

  std::cout << "[INIT] Threads deployed\n";

  return 0; 
}
