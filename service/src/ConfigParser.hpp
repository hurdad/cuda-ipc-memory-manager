#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <regex>
#include <string>
#include <fstream>
#include <cstdlib>

// flatbuffers schema
#include <flatbuffers/idl.h>
#include "service_bfbs_generated.h"

class ConfigParser {
public:
  static std::string envsubst(const std::string& input) {
    std::string result = input;

    // Regular expression to match environment variables, including default values
    std::regex re("\\$\\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\\}");

    // Search for all matches and replace with corresponding environment variables or default value
    std::smatch match;
    while (std::regex_search(result, match, re)) {
      std::string var_name      = match[1].str();
      std::string default_value = match[2].str(); // Default value (if any)

      const char* env_value = std::getenv(var_name.c_str());
      if (env_value) {
        result.replace(match.position(0), match.length(0), env_value);
      } else {
        // If no environment variable is found, use the default value
        result.replace(match.position(0), match.length(0),
                       default_value.empty() ? "" : default_value);
      }
    }

    return result;
  }

  static const std::vector<uint8_t> ParseServiceConfiguration(const std::string& json_file) {
    std::ifstream file(json_file);
    if (!file.is_open()) {
      throw std::runtime_error(fmt::format("Error: Unable to open file: {}", json_file));
    }

    // Read file content
    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();

    // Perform environment variable substitution
    std::string substituted_json = envsubst(json_content);
    spdlog::debug(substituted_json);

    // Get the embedded binary schema
    const uint8_t* bfbs_data =
        fbs::cuda::ipc::service::ConfigurationBinarySchema::data();
    size_t bfbs_size = fbs::cuda::ipc::service::ConfigurationBinarySchema::size();

    // Init parser and load schema
    flatbuffers::Parser parser;
    if (!parser.Deserialize(bfbs_data, bfbs_size)) {
      throw std::runtime_error("Configuration failed to parse binary schema");
    }
    spdlog::debug("Configuration binary schema parsed successfully!");

    // Parse json to fbs binary
    if (!parser.ParseJson(substituted_json.c_str())) {
      throw std::runtime_error("Configuration JSON parse failure!");
    }

    // Copy binary buffer to std::vector<uint8_t>
    std::vector<uint8_t> binary_buffer(parser.builder_.GetSize());
    memcpy(binary_buffer.data(), parser.builder_.GetBufferPointer(), parser.builder_.GetSize());
    return binary_buffer;
  }
};

#endif  // CONFIG_PARSER_H