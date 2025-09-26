/*
 *
 * boids_simple.cpp  Andrew Belles  Sept 26th, 2025 
 *
 * Simple boids 2D simulation aiming to show independent agents communicating via mesh 
 * to determine alignment, cohesion, and separation benchmarks for linear case 
 */ 

#include <chrono>
#include <stop_token>
#include <thread> 
#include <condition_variable>
#include <iostream> 
#include <vector> 
#include <atomic> 
#include <cmath> 
#include <algorithm>

using namespace std::chrono_literals;

constexpr double eps = 1e-9; 
constexpr double dt  = 1e-3;
constexpr double min_dist     = 0.3;
constexpr double w_seperation = 1.5; 
constexpr double w_alignment  = 1.0; 
constexpr double w_cohesion   = 0.8;
constexpr double max_accel    = 4.0; 
constexpr double max_vel      = 2.0; 

struct vec3 {
  double x, y, z; 

  /********** norms *****************************************/ 
  static double L1(const vec3 a, const vec3 b) {
    return (std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z));
  }

  static double L2(const vec3 a, const vec3 b) {
    const vec3 d = {a.x - b.x, a.y - b.y, a.z - b.z};
    return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  }

  /********** misc *****************************************/ 
  static vec3 clamp(const vec3 a, double lo, double hi)
  {
    vec3 r; 
    r.x = (a.x > hi) ? hi : (a.x < lo) ? lo : a.x; 
    r.y = (a.y > hi) ? hi : (a.y < lo) ? lo : a.y; 
    r.z = (a.z > hi) ? hi : (a.z < lo) ? lo : a.z; 
    return r; 
  }

  /********** overloaded functions *************************/

  vec3& operator+=(const vec3 b) {
    x += b.x; 
    y += b.y; 
    z += b.z; 
    return *this; 
  }

  vec3& operator/=(const double c) {
    x /= c; 
    y /= c; 
    z /= c; 
    return *this; 
  }

  friend vec3 operator+(const vec3 a, const vec3 b) 
  { 
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }
  friend vec3 operator-(const vec3 a, const vec3 b)
  { 
    return {a.x - b.x, a.y - b.y, a.z - b.z}; 
  }
  friend vec3 operator*(const double c, const vec3 a)
  {
    return {a.x * c, a.y * c, a.z * c}; 
  }
  friend vec3 operator/(const vec3 a, const double c)
  {
    return {a.x / c, a.y / c, a.z / c}; 
  }
}; 

constexpr vec3 initials[8] = {
  {0.0, 0.0, 0.0},
  {0.0, 9.0, 0.0},
  {3.0, 3.0, 0.0},
  {3.0, 6.0, 0.0},
  {6.0, 3.0, 0.0},
  {6.0, 6.0, 0.0},
  {9.0, 0.0, 0.0},
  {9.0, 9.0, 0.0}
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

  std::mutex m_; 
  std::condition_variable cv_; 
  std::atomic<size_t> epoch_{0};

public: 

  explicit Pool(size_t capacity=8) 
    : capacity_(capacity), 
      bufr_{new Packet[capacity], new Packet[capacity]}, 
      widx_{0, 0},
      active_(0),
      count_{0, 0},
      readers_{0, 0} {}

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
    readers_[prev].store(agent_count, std::memory_order_release);
    widx_[next].store(0, std::memory_order_release);

    {
      std::lock_guard lock(m_);
      epoch_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_all();
  }

  std::pair<size_t, std::vector<Packet>> 
  fetch(size_t seen_epoch)
  {
    {
      std::unique_lock lock(m_);
      cv_.wait(
        lock, 
        [&]{ return epoch_.load(std::memory_order_acquire) > seen_epoch; }
      );
    }

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
    size_t new_epoch = epoch_.load(std::memory_order_acquire);
    return {new_epoch, out}; 
  }
};

class SwarmAgent {
private: 
  Agent id_{0};  
  vec3 position_{0.0, 0.0, 0.0};
  vec3 velocity_{0.0, 0.0, 0.0};
  Pool& pool_; 

  void 
  boids_(std::vector<Packet> packets)
  {
    // Proximity filter 
    std::vector<std::tuple<vec3, vec3, double>> state;
    for (auto& [id, pos, vel] : packets) {
      if ( id == id_ ) {
        continue; 
      }

      // cache l2 distance in state tuple 
      double l2_dist = vec3::L2(pos, position_);
      if ( l2_dist > 5 ) {
        continue; 
      }

      state.push_back(std::make_tuple(pos, vel, l2_dist));
    } 
    size_t n(state.size()); 
    
    // compute separation, alignment, and cohesion 
    vec3 s{0.0, 0.0, 0.0}, a{0.0, 0.0, 0.0}, c{0.0, 0.0, 0.0};
    for (auto& [pos, vel, dist] : state) {
      s += ( (position_ - pos) / ( dist + eps) ); 
      a += vel; 
      c += pos; 
    }
    a = (a / n) - velocity_;  
    c = (c / n) - position_;

    // integrate position 
    vec3 accel = w_seperation * s + w_alignment * a + w_cohesion * c; 
    accel = vec3::clamp(accel, -max_accel, max_accel);

    vec3 vnext = velocity_ + (dt * accel); 
    vnext = vec3::clamp(vnext, -max_vel, max_vel);
    position_ = position_ + (dt * vnext);
    velocity_ = vnext; 
  }

public: 

  SwarmAgent(Agent id, vec3 x0, Pool& pool) 
    : id_(id),
      position_(x0),
      velocity_({0.0, 0.0, 0.0}),
      pool_(pool) {}

  void 
  run(std::stop_token& st) 
  {
    size_t seen = 0; 
    while ( !st.stop_requested() ) {
      pool_.push(std::make_tuple(id_, position_, velocity_));

      auto [new_epoch, packets] = pool_.fetch(seen);
      seen = new_epoch;

      boids_(packets);
    }
  }

  Agent id() const { return id_; }
  vec3 pos() const { return position_; }
};


int main(void)
{
  Pool pool(8);
  std::vector<SwarmAgent> agents;
  agents.reserve(8);
  
  for (int i(0); i < 8; i++) {
    agents.emplace_back(SwarmAgent(i, initials[i], pool)); 
  }
  std::cout << "[INIT] Agents Created\n";

  // create pool thread 
  std::jthread pool_thread([&pool](std::stop_token& st) {
    auto next = std::chrono::steady_clock::now(); 
    constexpr auto period = std::chrono::milliseconds(50);
    while ( !st.stop_requested() ) {
      pool.rotate(8);
      next += period; 
      std::this_thread::sleep_until(next);
    }
  });

  std::cout << "[INIT] Pool Thread Instantiated\n";

  std::vector<std::jthread> agent_threads; 
  agent_threads.reserve(8); 
  for (int i(0); i < 8; i++) {
    agent_threads.emplace_back([&a = agents[i]](std::stop_token& st) {
      a.run(st);
    });
  }

  std::cout << "[INIT] Agents Instantiated\n";

  std::this_thread::sleep_for(2s);

  return 0; 
}
