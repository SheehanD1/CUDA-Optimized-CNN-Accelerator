#pragma once

// ============================================================================
// MNIST Data Loader — Binary IDX Format Parser
// ============================================================================
//
// Parses the standard MNIST IDX file format:
//   - t10k-images-idx3-ubyte / train-images-idx3-ubyte (images)
//   - t10k-labels-idx1-ubyte / train-labels-idx1-ubyte (labels)
//
// IDX format (big-endian):
//   Images: [magic=2051][num_images][rows][cols][pixel_data...]
//   Labels: [magic=2049][num_labels][label_data...]
//
// Pixel values are normalized from [0, 255] to [0.0, 1.0].
// Images are returned as Tensor with shape (N, 1, 28, 28) in NCHW layout.
// Labels are returned as std::vector<int>.
//
// ============================================================================

#include "tensor.h"

#include <string>
#include <vector>

/// Load MNIST images from an IDX3-UBYTE file.
///
/// @param filepath Path to the MNIST image file (e.g., "t10k-images-idx3-ubyte")
/// @param max_images Maximum number of images to load (0 = load all)
/// @return Tensor of shape (N, 1, 28, 28) with pixel values in [0, 1]
///
Tensor load_mnist_images(const std::string& filepath, int max_images = 0);

/// Load MNIST labels from an IDX1-UBYTE file.
///
/// @param filepath Path to the MNIST label file (e.g., "t10k-labels-idx1-ubyte")
/// @param max_labels Maximum number of labels to load (0 = load all)
/// @return Vector of integer labels, each in [0, 9]
///
std::vector<int> load_mnist_labels(const std::string& filepath, int max_labels = 0);
