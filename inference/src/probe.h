#pragma once

#include <string>
#include <vector>
#include <utility>

// Forward declaration so we don't need to include all of ggml.h here
struct ggml_tensor;

class ProbeRegistry {
public:
    // Path to the folder you just generated with Python. 
    // Adjust this if you run your executable from a different relative directory!
    std::string golden_dir = "../golden_tensors/";

    // Saves the pointer to the tensor for auditing later
    void add_probe(const std::string& filename, struct ggml_tensor* tensor);

    // Iterates through the probes, reads the .bin files, and calculates MSE.
    // Returns true if the entire network passes, false if the kill-switch is tripped.
    bool audit() const;

private:
    // A simple list preserving the exact order the probes were added
    std::vector<std::pair<std::string, struct ggml_tensor*>> probes;
};