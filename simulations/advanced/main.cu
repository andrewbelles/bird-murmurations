/*
 * main.cu  Andrew Belles  Oct 1st, 2025 
 *
 * Main entry point for complex gpu boids simulation 
 *
 *
 */ 

#include "network/mailbox.cuh"
#include "simulation.cuh"
#include "agent/environment.cuh"
#include "network/communication.cuh"

#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h> 
#include <iostream> 
#include <fstream> 
#include <vector>
#include <vector_functions.h>
#include <unistd.h> 

namespace {

static inline float3 
operator*(float c, const float3 a)
{
  return make_float3(c * a.x, c * a.y, c * a.z);
}

static inline float3 
operator-(const float3 a, float c)
{
  return make_float3(a.x - c, a.y - c, a.z - c);
}

}

struct Args {
  // required 
  std::string env_file; 
  std::string sim_file; 
  std::string logger_file; 

  // optional 
  int agents      = 2048; 
  int bufr_size   = 32; 
  uint64_t epochs = 5000; 

  bool enable_loss  = false; 
  bool enable_noise = false; 
  float loss  = 0.0;
  float noise = 0.0; 
};

static void print_usage(const char* program);
static void parse_args(int argc, char* argv[], Args& args);
static void get_general_parameters(const Args& args, SimulationParams *sp, 
                                   EnvironmentParams *ep);
static void get_communication_parameters(const Args& args, float dist, 
                                         comms::Parameters* params);
static inline float3 random_vector();
static void initialize_state(const Args& args, std::vector<float3>& h_pos, 
                             std::vector<float3>& h_vel);


int main(int argc, char* argv[]) {
  cudaError_t status; 
  std::srand(getpid());
  Args args; 

  parse_args(argc, argv, args);
  std::cout << "[INIT] Arguments parsed correctly\n";

  SimulationParams sim_params; 
  EnvironmentParams env_params; 
  comms::Parameters comm_params; 
  
  get_general_parameters(args, &sim_params, &env_params);
  get_communication_parameters(args, sim_params.min_dist, &comm_params);
  std::cout << "[INIT] Parameters set\n";

  std::vector<float3> h_pos(args.agents), h_vel(args.agents);
  initialize_state(args, h_pos, h_vel);
  std::cout << "[INIT] State initialized\n";

  sim::Simulation simulation; 
  if ( (status = sim::create(
          &simulation, h_pos.data(), reinterpret_cast<const float*>(h_vel.data()), 
          args.agents, args.bufr_size, args.logger_file, &sim_params, &env_params)) 
        != cudaSuccess ) {
    std::cerr << "[ERROR] Failure in simulation initialization: ";
    std::cerr << cudaGetErrorString(status) << '\n';
    exit( 99 );
  }

  std::cout << "[INIT] Simulation initialized\n";

  for (uint64_t epoch(0); epoch < args.epochs; epoch++) {
    if ( (status = sim::step(&simulation, epoch, comm_params)) != cudaSuccess ) {
      sim::destroy(&simulation);
      exit( 99 ); 
    }
  }

  std::cout << "[SIM] Simulation ended\n";

  sim::destroy(&simulation);
  cudaDeviceSynchronize(); 
  return 0; 
}

static void 
initialize_state(const Args& args, std::vector<float3>& h_pos, 
                 std::vector<float3>& h_vel)
{
  for (int i = 0; i < args.agents; i++) {
    h_pos[i] = 2.0 * random_vector()- 1.0;
    h_vel[i] = random_vector();
  }
}

static void 
get_communication_parameters(const Args& args, float dist, comms::Parameters* params)
{
  if ( !params ) {
    std::cerr << "[ERROR] Invalid pointer passed to get_communication_parameters\n";
    exit( 99 );
  }

  params->radio_radius = dist * 5.0;  
  params->budget = 64; 
  params->policy = network::Overflow::Drop; 
  params->epoch  = 0; 
  params->enable_loss = args.enable_loss; 
  params->loss   = args.loss; 
  params->enable_noise = args.enable_noise; 
  params->noise  = args.noise; 
}


static void 
get_general_parameters(const Args& args, SimulationParams *sp, EnvironmentParams *ep)
{
  cudaError_t status; 
  if ( !sp || !ep ) {
    std::cerr << "[ERROR] Invalid pointers passed to get_general_parameters\n"; 
    exit( 99 );
  }

  if ( (status = read_environment_params(args.env_file.c_str(), ep)) != cudaSuccess ) {
    std::cerr << "[ERROR] Failed to read environment params file correctly ";
    std::cerr << cudaGetErrorString(status) << '\n';
    exit( 99 ); 
  }

  if ( (status = read_simulation_params(args.sim_file.c_str(), sp)) != cudaSuccess ) {
    std::cerr << "[ERROR] Failed to read simulation params file correctly ";
    std::cerr << cudaGetErrorString(status) << '\n';
    exit( 99 ); 
  }
}

static void
parse_args(int argc, char* argv[], Args& args)
{
    // Helper to print usage and terminate on error
    auto fail = [&](const std::string& msg) {
        std::cerr << "[ERROR] " << msg << "\n\n";
        print_usage(argv[0]);
        std::exit(99);
    };

    // Iterate through arguments. Flags must be followed by a value.
    for (int i = 1; i < argc; ++i) {
      std::string opt = argv[i];

      if (opt == "--env") {
        if (i + 1 >= argc) {
          fail("Missing value for --env");
        }
        args.env_file = argv[++i];
      } else if (opt == "--sim") {
        if (i + 1 >= argc) {
          fail("Missing value for --sim");
        }
        args.sim_file = argv[++i];
      } else if (opt == "--logger") {
        if (i + 1 >= argc) {
          fail("Missing value for --logger");
        }
        args.logger_file = argv[++i];
      } else if (opt == "--agents") {
        if (i + 1 >= argc) {
          fail("Missing value for --agents");
        }
        int n = std::stoi(argv[++i]);
        if (n <= 0) {
          fail("Number of agents must be positive");
        }
        args.agents = n;
      } else if (opt == "--bufr") {
        if (i + 1 >= argc) {
          fail("Missing value for --bufr");
        }
        int k = std::stoi(argv[++i]);
        if (k <= 0) {
          fail("Mailbox size must be positive");
        }
        args.bufr_size = k;
      } else if (opt == "--steps") {
        if (i + 1 >= argc) {
          fail("Missing value for --steps");
        }
        long long t = std::stoll(argv[++i]);
        if (t <= 0) {
          fail("Number of steps must be positive");
        }
        args.epochs = static_cast<uint64_t>(t);
      } else if (opt == "--loss") {
        if (i + 1 >= argc) {
          fail("Missing value for --loss");
        }
        float p = std::stof(argv[++i]);
        if (p < 0.0 || p > 1.0) {
          fail("Loss probability must be in [0,1]");
        }
        args.loss = p;
        args.enable_loss = (p > 0.0);
      } else if (opt == "--noise") {
        if (i + 1 >= argc) {
          fail("Missing value for --noise");
        }
        float s = std::stof(argv[++i]);
        if (s < 0.0) {
          fail("Noise magnitude must be non‑negative");
        }
        args.noise = s;
        args.enable_noise = (s > 0.0);
      } else {
        fail("Unknown option: " + opt);
      }
    }

    // Verify all required file arguments were provided
    if (args.env_file.empty() || args.sim_file.empty() || args.logger_file.empty()) {
        fail("--env, --sim, and --logger must all be specified");
    }

    // Verify that the environment and simulation files exist
    auto check_exists = [&](const std::string& path, const char* flag) {
        std::ifstream f(path);
        if (!f.is_open()) {
            fail(std::string("Invalid file path for ") + flag + ": " + path);
        }
    };
    check_exists(args.env_file, "--env");
    check_exists(args.sim_file, "--sim");
}

static void 
print_usage(const char* program)
{
  std::cerr << "Usage: " << program << " --env <env_file> --sim <sim_file>" 
            << " --logger <output_file>"
            << " [--agents N] [--bufr K] [--steps T] [--loss p] [--noise s]"
            << "  --env <path>      Path to environment parameter file\n"
            << "  --sim <path>      Path to simulation (boids) parameter file\n"
            << "  --logger <path>   Path to output log file for snapshots\n"
            << "  --agents N        Number of agents [default 2048]\n"
            << "  --bufr K          Mailbox capacity per agent [default 32]\n"
            << "  --steps T         Number of simulation epochs [default 5000]\n"
            << "  --loss p          Packet loss probability\n"
            << "  --noise s         Additive noise magnitude\n";
}

static inline float3 random_vector()
{
  return make_float3(
    static_cast<float>(std::rand()) / RAND_MAX,
    static_cast<float>(std::rand()) / RAND_MAX,
    static_cast<float>(std::rand()) / RAND_MAX
  );
}

