#include "probe.h"
#include "ggml.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>

void ProbeRegistry::add_probe(const std::string& filename, struct ggml_tensor* tensor) {
    probes.push_back({filename, tensor});
}

bool ProbeRegistry::audit() const {
    std::cout << "\n========================================================\n";
    std::cout << "               GOLDEN TENSOR AUDIT                      \n";
    std::cout << "========================================================\n";

    if (probes.empty()) {
        std::cout << "[INFO] No probes registered.\n";
        return true;
    }

    bool all_passed = true;
    
    // FP16 drift tolerance. PyTorch and GGML handle FP16 epsilon rounding 
    // slightly differently, so it will never be a perfect 0.000000.
    const double TOLERANCE = 5e-2;

    for (const auto& probe : probes) {
        const std::string& filename = probe.first;
        struct ggml_tensor* tensor = probe.second;

        std::string full_path = golden_dir + filename;
        std::ifstream file(full_path, std::ios::binary);
        
        if (!file) {
            std::cerr << "[AUDIT] " << filename << " : [ERROR] Could not open file.\n";
            all_passed = false;
            break; // KILL SWITCH
        }

        // 1. Read the header (The number of elements in the Python tensor)
        int32_t expected_elements = 0;
        file.read(reinterpret_cast<char*>(&expected_elements), sizeof(int32_t));

        // 2. Validate Tensor Geometry
        int64_t actual_elements = ggml_nelements(tensor);
        if (expected_elements != actual_elements) {
            std::cerr << "[AUDIT] " << filename << " : [FAIL] Shape mismatch. Expected " 
                      << expected_elements << " floats, got " << actual_elements << ".\n";
            all_passed = false;
            break; // KILL SWITCH
        }

        // 3. Read the Golden Floats from Disk
        std::vector<float> golden_data(expected_elements);
        file.read(reinterpret_cast<char*>(golden_data.data()), expected_elements * sizeof(float));

        // 4. Calculate Mean Squared Error (MSE) and Max Absolute Error
        float* cpp_data = (float*)tensor->data;
        double mse = 0.0;
        double max_err = 0.0;

      if (tensor->type == GGML_TYPE_F32) {
            float* cpp_data = (float*)tensor->data;
            for (int64_t i = 0; i < actual_elements; ++i) {
                double diff = static_cast<double>(cpp_data[i]) - static_cast<double>(golden_data[i]);
                mse += diff * diff;
                if (std::abs(diff) > max_err) {
                    max_err = std::abs(diff);
                }
            }
        } else if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_t* cpp_data = (ggml_fp16_t*)tensor->data;
            for (int64_t i = 0; i < actual_elements; ++i) {
                // Safely convert the 16-bit float back to 32-bit before doing math
                float val = ggml_fp16_to_fp32(cpp_data[i]);
                double diff = static_cast<double>(val) - static_cast<double>(golden_data[i]);
                mse += diff * diff;
                if (std::abs(diff) > max_err) {
                    max_err = std::abs(diff);
                }
            }
        } else {
            std::cerr << "[AUDIT] Unsupported tensor type for audit.\n";
            all_passed = false;
            break;
        }
        mse /= actual_elements;

        // 5. Evaluate and Print
        if (mse <= TOLERANCE) {
            printf("[AUDIT] %-20s : [PASS] (MSE: %.6f, MaxErr: %.6f)\n", filename.c_str(), mse, max_err);
        } else {
            printf("[AUDIT] %-20s : [FAIL] (MSE: %.6f, MaxErr: %.6f) <-- KILL SWITCH\n", filename.c_str(), mse, max_err);
            all_passed = false;
            break; // KILL SWITCH: Stop testing. Subsequent layers are guaranteed to fail.
        }
    }

    std::cout << "========================================================\n";
    return all_passed;
}