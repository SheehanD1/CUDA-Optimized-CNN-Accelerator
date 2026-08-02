#include "inference.h"
#include "model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ============================================================================
// Model Construction Tests
// ============================================================================

TEST(Model, DefaultShapes) {
    Model model;

    EXPECT_EQ(model.conv1_weights.shape(), (std::vector<int>{8, 1, 3, 3}));
    EXPECT_EQ(model.conv1_bias.shape(), (std::vector<int>{8}));
    EXPECT_EQ(model.conv2_weights.shape(), (std::vector<int>{16, 8, 3, 3}));
    EXPECT_EQ(model.conv2_bias.shape(), (std::vector<int>{16}));
    EXPECT_EQ(model.dense1_weights.shape(), (std::vector<int>{120, 784}));
    EXPECT_EQ(model.dense1_bias.shape(), (std::vector<int>{120}));
    EXPECT_EQ(model.dense2_weights.shape(), (std::vector<int>{10, 120}));
    EXPECT_EQ(model.dense2_bias.shape(), (std::vector<int>{10}));
}

TEST(Model, DefaultZeroInitialized) {
    Model model;

    // All tensors should be zero-initialized
    for (int i = 0; i < model.conv1_weights.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(model.conv1_weights[i], 0.0f);
    }
    for (int i = 0; i < model.dense2_bias.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(model.dense2_bias[i], 0.0f);
    }
}

TEST(Model, TotalParameters) {
    Model model;

    // Conv1: 8*1*3*3 + 8 = 72 + 8 = 80
    // Conv2: 16*8*3*3 + 16 = 1152 + 16 = 1168
    // Dense1: 120*784 + 120 = 94080 + 120 = 94200
    // Dense2: 10*120 + 10 = 1200 + 10 = 1210
    // Total: 80 + 1168 + 94200 + 1210 = 96658
    EXPECT_EQ(model.total_parameters(), 96658);
}

// ============================================================================
// Xavier Initialization Tests
// ============================================================================

TEST(ModelXavier, InitializesNonZero) {
    Model model;
    model.initialize_xavier(42);

    // Weights should not all be zero after Xavier init
    bool has_nonzero = false;
    for (int i = 0; i < model.conv1_weights.num_elements(); ++i) {
        if (model.conv1_weights[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(ModelXavier, BiasesRemainZero) {
    Model model;
    model.initialize_xavier(42);

    // All biases should remain zero
    for (int i = 0; i < model.conv1_bias.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(model.conv1_bias[i], 0.0f);
    }
    for (int i = 0; i < model.conv2_bias.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(model.conv2_bias[i], 0.0f);
    }
    for (int i = 0; i < model.dense1_bias.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(model.dense1_bias[i], 0.0f);
    }
    for (int i = 0; i < model.dense2_bias.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(model.dense2_bias[i], 0.0f);
    }
}

TEST(ModelXavier, ValuesInExpectedRange) {
    Model model;
    model.initialize_xavier(42);

    // Conv1: fan_in=9, fan_out=72 → limit = sqrt(6/81) ≈ 0.272
    float conv1_limit = std::sqrt(6.0f / (9.0f + 72.0f));
    for (int i = 0; i < model.conv1_weights.num_elements(); ++i) {
        EXPECT_GE(model.conv1_weights[i], -conv1_limit);
        EXPECT_LE(model.conv1_weights[i], conv1_limit);
    }

    // Dense2: fan_in=120, fan_out=10 → limit = sqrt(6/130) ≈ 0.215
    float dense2_limit = std::sqrt(6.0f / (120.0f + 10.0f));
    for (int i = 0; i < model.dense2_weights.num_elements(); ++i) {
        EXPECT_GE(model.dense2_weights[i], -dense2_limit);
        EXPECT_LE(model.dense2_weights[i], dense2_limit);
    }
}

TEST(ModelXavier, DeterministicWithSameSeed) {
    Model model1;
    model1.initialize_xavier(123);

    Model model2;
    model2.initialize_xavier(123);

    // Same seed should produce identical weights
    EXPECT_TRUE(model1.conv1_weights == model2.conv1_weights);
    EXPECT_TRUE(model1.conv2_weights == model2.conv2_weights);
    EXPECT_TRUE(model1.dense1_weights == model2.dense1_weights);
    EXPECT_TRUE(model1.dense2_weights == model2.dense2_weights);
}

TEST(ModelXavier, DifferentSeedsDiffer) {
    Model model1;
    model1.initialize_xavier(42);

    Model model2;
    model2.initialize_xavier(99);

    // Different seeds should produce different weights
    EXPECT_FALSE(model1.conv1_weights == model2.conv1_weights);
}

// ============================================================================
// Save/Load Serialization Tests
// ============================================================================

TEST(ModelIO, SaveLoadRoundTrip) {
    Model original;
    original.initialize_xavier(42);

    // Save to temporary file
    std::string filepath = "test_model_roundtrip.bin";
    original.save(filepath);

    // Load into a new model
    Model loaded;
    loaded.load(filepath);

    // Verify all tensors match exactly
    EXPECT_TRUE(original.conv1_weights == loaded.conv1_weights);
    EXPECT_TRUE(original.conv1_bias == loaded.conv1_bias);
    EXPECT_TRUE(original.conv2_weights == loaded.conv2_weights);
    EXPECT_TRUE(original.conv2_bias == loaded.conv2_bias);
    EXPECT_TRUE(original.dense1_weights == loaded.dense1_weights);
    EXPECT_TRUE(original.dense1_bias == loaded.dense1_bias);
    EXPECT_TRUE(original.dense2_weights == loaded.dense2_weights);
    EXPECT_TRUE(original.dense2_bias == loaded.dense2_bias);

    // Clean up
    std::remove(filepath.c_str());
}

TEST(ModelIO, LoadNonexistentThrows) {
    Model model;
    EXPECT_THROW(model.load("nonexistent_file.bin"), std::runtime_error);
}

// ============================================================================
// Pipeline Smoke Test — Xavier-initialized model produces valid output
// ============================================================================

TEST(ModelPipeline, SmokeTestProducesValidProbabilities) {
    Model model;
    model.initialize_xavier(42);

    // Create a fake MNIST-like input (1 sample, 1 channel, 28x28)
    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);

    // Run full inference
    Tensor output = cpu_inference(model, input);

    // Output shape should be (1, 10)
    EXPECT_EQ(output.shape(), (std::vector<int>{1, 10}));

    // All probabilities should be in [0, 1]
    for (int j = 0; j < 10; ++j) {
        EXPECT_GE(output.at(0, j), 0.0f) << "Negative probability at class " << j;
        EXPECT_LE(output.at(0, j), 1.0f) << "Probability > 1 at class " << j;
    }

    // Probabilities should sum to ~1.0
    float sum = 0.0f;
    for (int j = 0; j < 10; ++j) {
        sum += output.at(0, j);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(ModelPipeline, PredictReturnsValidClass) {
    Model model;
    model.initialize_xavier(42);

    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);
    std::vector<int> predictions = cpu_predict(model, input);

    EXPECT_EQ(predictions.size(), 1u);
    EXPECT_GE(predictions[0], 0);
    EXPECT_LE(predictions[0], 9);
}

TEST(ModelPipeline, DeterministicInference) {
    Model model;
    model.initialize_xavier(42);

    Tensor input = Tensor::rand({1, 1, 28, 28}, 42);

    Tensor output1 = cpu_inference(model, input);
    Tensor output2 = cpu_inference(model, input);

    // Same model + same input → identical output
    EXPECT_TRUE(output1 == output2);
}
