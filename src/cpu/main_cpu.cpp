// ============================================================================
// CPU Inference Executable — MNIST Digit Classification
// ============================================================================
//
// Usage:
//   ./cnn_cpu <weights_file> <images_file> <labels_file> [num_images]
//
// Example:
//   ./cnn_cpu weights/model.bin data/t10k-images-idx3-ubyte \
//             data/t10k-labels-idx1-ubyte 10
//
// If no weights file exists, initializes with Xavier and saves.
// ============================================================================

#include "data_loader.h"
#include "inference.h"
#include "model.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <weights_file> <images_file> <labels_file> [num_images]\n"
              << "\n"
              << "Arguments:\n"
              << "  weights_file  Path to model weights (.bin)\n"
              << "                If file doesn't exist, initializes with Xavier and saves.\n"
              << "  images_file   Path to MNIST images (IDX3-UBYTE format)\n"
              << "  labels_file   Path to MNIST labels (IDX1-UBYTE format)\n"
              << "  num_images    Number of images to classify (default: 10)\n";
}

static void print_prediction(int index, const Tensor& probs, int predicted,
                              int label) {
    std::cout << "  Image " << std::setw(5) << index
              << " | Predicted: " << predicted
              << " | Actual: " << label
              << " | Confidence: " << std::fixed << std::setprecision(4)
              << (probs.at(0, predicted) * 100.0f) << "%"
              << (predicted == label ? " ✓" : " ✗") << "\n";
}

int main(int argc, char* argv[]) {
    // ========================================================================
    // Parse arguments
    // ========================================================================

    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string weights_path = argv[1];
    std::string images_path = argv[2];
    std::string labels_path = argv[3];
    int num_images = (argc >= 5) ? std::stoi(argv[4]) : 10;

    // ========================================================================
    // Load or initialize model
    // ========================================================================

    std::cout << "====================================================\n";
    std::cout << "  CNN CPU Inference — MNIST Digit Classification\n";
    std::cout << "====================================================\n\n";

    Model model;

    // Check if weights file exists
    {
        std::ifstream check(weights_path, std::ios::binary);
        if (check.good()) {
            std::cout << "Loading model weights from: " << weights_path << "\n";
            model.load(weights_path);
        } else {
            std::cout << "Weights file not found. Initializing with Xavier...\n";
            model.initialize_xavier(42);
            model.save(weights_path);
            std::cout << "Saved initialized weights to: " << weights_path << "\n";
            std::cout << "NOTE: Untrained model — predictions will be random.\n";
        }
    }

    model.print_summary();
    std::cout << "\n";

    // ========================================================================
    // Load MNIST data
    // ========================================================================

    std::cout << "Loading MNIST images from: " << images_path << "\n";
    std::cout << "Loading MNIST labels from: " << labels_path << "\n";
    std::cout << "Number of images: " << num_images << "\n\n";

    Tensor images = load_mnist_images(images_path, num_images);
    std::vector<int> labels = load_mnist_labels(labels_path, num_images);

    // ========================================================================
    // Run inference
    // ========================================================================

    std::cout << "Running inference...\n";
    std::cout << "----------------------------------------------------\n";

    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        // Extract single image: slice from batch
        // Images tensor is (N, 1, 28, 28), extract image i as (1, 1, 28, 28)
        Tensor single_image({1, 1, 28, 28});
        int offset = i * 1 * 28 * 28;
        for (int p = 0; p < 1 * 28 * 28; ++p) {
            single_image[p] = images[offset + p];
        }

        // Run inference on single image
        Tensor probs = cpu_inference(model, single_image);
        int predicted = probs.argmax_per_row()[0];

        print_prediction(i, probs, predicted, labels[static_cast<size_t>(i)]);

        if (predicted == labels[static_cast<size_t>(i)]) {
            ++correct;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // ========================================================================
    // Print results
    // ========================================================================

    std::cout << "----------------------------------------------------\n";
    std::cout << "\nResults:\n";
    std::cout << "  Accuracy:   " << correct << "/" << num_images
              << " (" << std::fixed << std::setprecision(2)
              << (100.0f * static_cast<float>(correct) / static_cast<float>(num_images))
              << "%)\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(2)
              << elapsed_ms << " ms\n";
    std::cout << "  Per image:  " << std::fixed << std::setprecision(2)
              << (elapsed_ms / static_cast<double>(num_images)) << " ms\n";
    std::cout << "====================================================\n";

    return 0;
}
