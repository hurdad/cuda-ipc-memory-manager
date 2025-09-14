#include <iostream>
#include <csignal>
#include <boost/program_options.hpp>


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

  return 0;
}