#include "model.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

// ============================================================================
// Helper: write/read a tensor to/from a binary stream
// ============================================================================

static void write_tensor(std::ofstream& out, const Tensor& t) {
    out.write(reinterpret_cast<const char*>(t.data()),
              static_cast<std::streamsize>(t.num_elements()) *
              static_cast<std::streamsize>(sizeof(float)));
}

static void read_tensor(std::ifstream& in, Tensor& t) {
    in.read(reinterpret_cast<char*>(t.data()),
            static_cast<std::streamsize>(t.num_elements()) *
            static_cast<std::streamsize>(sizeof(float)));
}

// ============================================================================
// Model Serialization
// ============================================================================

void Model::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Model::save: cannot open file '" + filepath + "'");
    }

    // Write all tensors in layer order
    write_tensor(out, conv1_weights);
    write_tensor(out, conv1_bias);
    write_tensor(out, conv2_weights);
    write_tensor(out, conv2_bias);
    write_tensor(out, dense1_weights);
    write_tensor(out, dense1_bias);
    write_tensor(out, dense2_weights);
    write_tensor(out, dense2_bias);

    if (!out.good()) {
        throw std::runtime_error("Model::save: error writing to '" + filepath + "'");
    }
}

void Model::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Model::load: cannot open file '" + filepath + "'");
    }

    // Read all tensors in the same order they were saved
    read_tensor(in, conv1_weights);
    read_tensor(in, conv1_bias);
    read_tensor(in, conv2_weights);
    read_tensor(in, conv2_bias);
    read_tensor(in, dense1_weights);
    read_tensor(in, dense1_bias);
    read_tensor(in, dense2_weights);
    read_tensor(in, dense2_bias);

    if (!in.good()) {
        throw std::runtime_error("Model::load: error reading from '" + filepath + "'");
    }
}

// ============================================================================
// Model Summary
// ============================================================================

void Model::print_summary() const {
    std::cout << "====================================================\n";
    std::cout << "CNN Model Summary (LeNet-style)\n";
    std::cout << "====================================================\n";
    std::cout << "Layer                Shape               Params\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << "Conv1 weights        (8, 1, 3, 3)        "
              << conv1_weights.num_elements() << "\n";
    std::cout << "Conv1 bias           (8)                  "
              << conv1_bias.num_elements() << "\n";
    std::cout << "Conv2 weights        (16, 8, 3, 3)       "
              << conv2_weights.num_elements() << "\n";
    std::cout << "Conv2 bias           (16)                 "
              << conv2_bias.num_elements() << "\n";
    std::cout << "Dense1 weights       (120, 784)           "
              << dense1_weights.num_elements() << "\n";
    std::cout << "Dense1 bias          (120)                "
              << dense1_bias.num_elements() << "\n";
    std::cout << "Dense2 weights       (10, 120)            "
              << dense2_weights.num_elements() << "\n";
    std::cout << "Dense2 bias          (10)                 "
              << dense2_bias.num_elements() << "\n";
    std::cout << "----------------------------------------------------\n";
    std::cout << "Total parameters:    " << total_parameters() << "\n";
    std::cout << "====================================================\n";
}
