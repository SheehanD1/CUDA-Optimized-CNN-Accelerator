#pragma once

// ============================================================================
// CNN Model — Holds all layer parameters for the LeNet-style architecture
// ============================================================================
//
// Architecture:
//   Conv2D(1→8, 3×3, pad=1) → ReLU → MaxPool(2×2)
//   → Conv2D(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2)
//   → Flatten → Dense(16*7*7=784, 120) → ReLU
//   → Dense(120, 10) → Softmax
//
// Input:  (N, 1, 28, 28) — MNIST grayscale images
// Output: (N, 10) — probability distribution over digits 0–9
//
// After Conv1+Pool: (N, 8, 14, 14)
// After Conv2+Pool: (N, 16, 7, 7)
// After Flatten:    (N, 784)
// After Dense1:     (N, 120)
// After Dense2:     (N, 10)
// ============================================================================

#include "tensor.h"

#include <random>
#include <string>

/// CNN model holding all trainable parameters (weights + biases).
struct Model {
    // ========================================================================
    // Layer Parameters
    // ========================================================================

    // Conv layer 1: 1 input channel → 8 output channels, 3×3 kernel
    Tensor conv1_weights;   // Shape: (8, 1, 3, 3)
    Tensor conv1_bias;      // Shape: (8)

    // Conv layer 2: 8 input channels → 16 output channels, 3×3 kernel
    Tensor conv2_weights;   // Shape: (16, 8, 3, 3)
    Tensor conv2_bias;      // Shape: (16)

    // Dense layer 1: 784 → 120
    Tensor dense1_weights;  // Shape: (120, 784)
    Tensor dense1_bias;     // Shape: (120)

    // Dense layer 2: 120 → 10
    Tensor dense2_weights;  // Shape: (10, 120)
    Tensor dense2_bias;     // Shape: (10)

    // ========================================================================
    // Construction
    // ========================================================================

    /// Default constructor: creates tensors with correct shapes, zero-initialized.
    Model()
        : conv1_weights({8, 1, 3, 3}),
          conv1_bias({8}),
          conv2_weights({16, 8, 3, 3}),
          conv2_bias({16}),
          dense1_weights({120, 784}),
          dense1_bias({120}),
          dense2_weights({10, 120}),
          dense2_bias({10}) {}

    // ========================================================================
    // Weight Initialization
    // ========================================================================

    /// Initialize weights using Xavier/Glorot uniform distribution.
    /// Biases are zero-initialized.
    ///
    /// Xavier uniform: W ~ Uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    ///
    /// @param seed Random seed for reproducible initialization.
    ///
    void initialize_xavier(unsigned int seed = 42) {
        std::mt19937 gen(seed);

        auto xavier_init = [&gen](Tensor& weights, int fan_in, int fan_out) {
            float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (int i = 0; i < weights.num_elements(); ++i) {
                weights[i] = dist(gen);
            }
        };

        // Conv1: fan_in = 1*3*3=9, fan_out = 8*3*3=72
        xavier_init(conv1_weights, 9, 72);
        conv1_bias.zero();

        // Conv2: fan_in = 8*3*3=72, fan_out = 16*3*3=144
        xavier_init(conv2_weights, 72, 144);
        conv2_bias.zero();

        // Dense1: fan_in = 784, fan_out = 120
        xavier_init(dense1_weights, 784, 120);
        dense1_bias.zero();

        // Dense2: fan_in = 120, fan_out = 10
        xavier_init(dense2_weights, 120, 10);
        dense2_bias.zero();
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Save model weights to a binary file.
    /// Format: each tensor stored as raw float32 values in order.
    void save(const std::string& filepath) const;

    /// Load model weights from a binary file.
    /// The model must already have correct shapes (default constructor).
    void load(const std::string& filepath);

    // ========================================================================
    // Info
    // ========================================================================

    /// Returns the total number of trainable parameters.
    int total_parameters() const {
        return conv1_weights.num_elements() + conv1_bias.num_elements() +
               conv2_weights.num_elements() + conv2_bias.num_elements() +
               dense1_weights.num_elements() + dense1_bias.num_elements() +
               dense2_weights.num_elements() + dense2_bias.num_elements();
    }

    /// Print model summary (layer shapes and parameter counts).
    void print_summary() const;
};
