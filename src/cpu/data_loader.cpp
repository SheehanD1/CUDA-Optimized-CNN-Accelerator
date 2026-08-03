#include "data_loader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

// ============================================================================
// Helper: read a big-endian 32-bit integer from a binary stream
// ============================================================================

static uint32_t read_uint32_be(std::ifstream& in) {
    uint8_t bytes[4];
    in.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           (static_cast<uint32_t>(bytes[3]));
}

// ============================================================================
// Load MNIST Images (IDX3-UBYTE format)
// ============================================================================
//
// File format:
//   [0000] magic number = 2051 (0x00000803)
//   [0004] number of images
//   [0008] number of rows (28)
//   [0012] number of columns (28)
//   [0016] pixel data: unsigned byte, row-major, values in [0, 255]
//

Tensor load_mnist_images(const std::string& filepath, int max_images) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            "load_mnist_images: cannot open file '" + filepath + "'");
    }

    // Read header
    uint32_t magic = read_uint32_be(in);
    if (magic != 2051) {
        throw std::runtime_error(
            "load_mnist_images: invalid magic number " + std::to_string(magic) +
            " (expected 2051). Is this an MNIST image file?");
    }

    uint32_t num_images = read_uint32_be(in);
    uint32_t rows = read_uint32_be(in);
    uint32_t cols = read_uint32_be(in);

    if (rows != 28 || cols != 28) {
        throw std::runtime_error(
            "load_mnist_images: unexpected image dimensions " +
            std::to_string(rows) + "x" + std::to_string(cols) +
            " (expected 28x28)");
    }

    // Limit number of images if requested
    if (max_images > 0 && static_cast<uint32_t>(max_images) < num_images) {
        num_images = static_cast<uint32_t>(max_images);
    }

    int n = static_cast<int>(num_images);
    int h = static_cast<int>(rows);
    int w = static_cast<int>(cols);
    int pixels_per_image = h * w;

    // Allocate output tensor: (N, 1, 28, 28) — single channel grayscale
    Tensor images({n, 1, h, w});

    // Read pixel data and normalize to [0, 1]
    std::vector<uint8_t> buffer(static_cast<size_t>(pixels_per_image));
    for (int i = 0; i < n; ++i) {
        in.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(pixels_per_image));

        if (!in.good()) {
            throw std::runtime_error(
                "load_mnist_images: unexpected end of file at image " +
                std::to_string(i));
        }

        for (int p = 0; p < pixels_per_image; ++p) {
            images[i * pixels_per_image + p] =
                static_cast<float>(buffer[static_cast<size_t>(p)]) / 255.0f;
        }
    }

    return images;
}

// ============================================================================
// Load MNIST Labels (IDX1-UBYTE format)
// ============================================================================
//
// File format:
//   [0000] magic number = 2049 (0x00000801)
//   [0004] number of labels
//   [0008] label data: unsigned byte, values in [0, 9]
//

std::vector<int> load_mnist_labels(const std::string& filepath, int max_labels) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            "load_mnist_labels: cannot open file '" + filepath + "'");
    }

    // Read header
    uint32_t magic = read_uint32_be(in);
    if (magic != 2049) {
        throw std::runtime_error(
            "load_mnist_labels: invalid magic number " + std::to_string(magic) +
            " (expected 2049). Is this an MNIST label file?");
    }

    uint32_t num_labels = read_uint32_be(in);

    // Limit number of labels if requested
    if (max_labels > 0 && static_cast<uint32_t>(max_labels) < num_labels) {
        num_labels = static_cast<uint32_t>(max_labels);
    }

    int n = static_cast<int>(num_labels);

    // Read label data
    std::vector<uint8_t> buffer(static_cast<size_t>(n));
    in.read(reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(n));

    if (!in.good()) {
        throw std::runtime_error(
            "load_mnist_labels: unexpected end of file");
    }

    // Convert to int vector
    std::vector<int> labels(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        labels[static_cast<size_t>(i)] = static_cast<int>(buffer[static_cast<size_t>(i)]);
    }

    return labels;
}
