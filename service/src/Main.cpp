#include <iostream>
#include <csignal>
#include <atomic>
#include <spdlog/spdlog.h> 				      // core spdlog library
#include <spdlog/cfg/env.h> 			      // for spdlog::cfg::load_env_levels()
#include <boost/program_options.hpp>

// config parser
#include "ConfigParser.hpp"
#include "service_generated.h"

// server implementation
#include "thread/CudaIPCServer.h"

static std::atomic_bool g_shutdown(false); // Flag to control server shutdown

// Signal handler for graceful exit
inline void SignalHandler(int signum) {
  spdlog::warn("Received signal: {}. Shutting down server..", signum);
  g_shutdown.store(true);
}

int main(int argc, char** argv) {
  // command line options
  std::string                                 config_file;
  boost::program_options::options_description desc("Options");
  desc.add_options()
      ("help,h", "Options related to the program.")
      ("config,c", boost::program_options::value<std::string>(&config_file)->required(), "Configuration File");

  boost::program_options::variables_map vm;
  try {
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
    //print help
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return EXIT_SUCCESS;
    }
    boost::program_options::notify(vm);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // load log levels from env variable (SPDLOG_LEVEL=debug) info is default
  spdlog::cfg::load_env_levels();

  // Register signal handlers for Ctrl+C (SIGINT) and (SIGTERM)
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  // parse Configuration to fbs
  const std::vector<uint8_t> config_binary = ConfigParser::ParseServiceConfiguration(config_file); // must stay in scope
  auto config = fbs::cuda::ipc::service::GetConfiguration(config_binary.data());

  // init cuda ipc server
  CudaIPCServer server(config->zmq_router_endpoint()->str());
  server.start();

  // wait until shutdown
  while (!g_shutdown.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep to avoid busy loop
  }

  // clean up server
  server.stop();
  server.join();

  spdlog::info("Server shutdown complete.");
  spdlog::shutdown();

  return 0;
}