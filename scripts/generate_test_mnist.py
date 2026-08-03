#!/usr/bin/env python3
"""Generate synthetic MNIST IDX files for unit testing.

Creates small test files that match the real MNIST IDX binary format
but contain only a few images/labels. This allows unit tests to validate
the data loader without requiring the full MNIST dataset download.

Usage:
    python scripts/generate_test_mnist.py
"""

import struct
import os


def write_uint32_be(f, value):
    """Write a big-endian 32-bit unsigned integer."""
    f.write(struct.pack('>I', value))


def generate_test_images(filepath, num_images=5, rows=28, cols=28, seed=42):
    """Generate a synthetic MNIST image file in IDX3-UBYTE format."""
    import random
    random.seed(seed)

    with open(filepath, 'wb') as f:
        # Header
        write_uint32_be(f, 2051)        # Magic number for images
        write_uint32_be(f, num_images)
        write_uint32_be(f, rows)
        write_uint32_be(f, cols)

        # Pixel data: uint8 values
        for i in range(num_images):
            pixels = bytes([random.randint(0, 255) for _ in range(rows * cols)])
            f.write(pixels)

    print(f"Generated {filepath}: {num_images} images ({rows}x{cols})")


def generate_test_labels(filepath, num_labels=5, seed=42):
    """Generate a synthetic MNIST label file in IDX1-UBYTE format."""
    import random
    random.seed(seed)

    # Use a deterministic label sequence
    labels = [i % 10 for i in range(num_labels)]

    with open(filepath, 'wb') as f:
        # Header
        write_uint32_be(f, 2049)        # Magic number for labels
        write_uint32_be(f, num_labels)

        # Label data: uint8 values in [0, 9]
        f.write(bytes(labels))

    print(f"Generated {filepath}: {num_labels} labels = {labels}")


if __name__ == '__main__':
    # Create test data directory
    test_data_dir = os.path.join('tests', 'data')
    os.makedirs(test_data_dir, exist_ok=True)

    # Generate small test files
    generate_test_images(
        os.path.join(test_data_dir, 'test-images-idx3-ubyte'),
        num_images=5
    )
    generate_test_labels(
        os.path.join(test_data_dir, 'test-labels-idx1-ubyte'),
        num_labels=5
    )

    print("\nTest MNIST files generated in tests/data/")
    print("These are synthetic files for unit testing only.")
