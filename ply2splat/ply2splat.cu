// File: Ply2Splat.cu

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "tinyply.h"

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Structure to store vertex data in memory
struct Vertex
{
    float x, y, z;
    float scale0, scale1, scale2;
    float rot0, rot1, rot2, rot3;
    float f_dc_0, f_dc_1, f_dc_2;
    float opacity;
};

// Structure for output vertex data
struct OutputVertex
{
    float position[3];
    float scales[3];
    uint8_t color[4];
    uint8_t rot[4];
};

// Constant for SH_C0
__constant__ double d_SH_C0;
static const double SH_C0 = 0.28209479177387814;

// CUDA kernel for per-vertex computations
__global__ void compute_vertices_kernel(const Vertex* d_vertices, OutputVertex* d_output, float* d_sort_keys, size_t num_vertices, double SH_C0)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    // Read input vertex
    Vertex v = d_vertices[idx];

    // Compute position
    d_output[idx].position[0] = v.x;
    d_output[idx].position[1] = v.y;
    d_output[idx].position[2] = v.z;

    // Compute scales
    d_output[idx].scales[0] = expf(v.scale0);
    d_output[idx].scales[1] = expf(v.scale1);
    d_output[idx].scales[2] = expf(v.scale2);

    // Compute color
    float color_f0 = 0.5f + static_cast<float>(SH_C0) * v.f_dc_0;
    float color_f1 = 0.5f + static_cast<float>(SH_C0) * v.f_dc_1;
    float color_f2 = 0.5f + static_cast<float>(SH_C0) * v.f_dc_2;
    float denominator = 1.0f + expf(-v.opacity);
    float alpha = 1.0f / denominator;

    // Clamp and convert to [0, 255]
    d_output[idx].color[0] = static_cast<uint8_t>(fminf(fmaxf(color_f0 * 255.0f, 0.0f), 255.0f));
    d_output[idx].color[1] = static_cast<uint8_t>(fminf(fmaxf(color_f1 * 255.0f, 0.0f), 255.0f));
    d_output[idx].color[2] = static_cast<uint8_t>(fminf(fmaxf(color_f2 * 255.0f, 0.0f), 255.0f));
    d_output[idx].color[3] = static_cast<uint8_t>(fminf(fmaxf(alpha * 255.0f, 0.0f), 255.0f));

    // Normalize and transform rotation
    float norm = sqrtf(v.rot0 * v.rot0 + v.rot1 * v.rot1 + v.rot2 * v.rot2 + v.rot3 * v.rot3);
    norm = (norm < 1e-12f) ? 1e-12f : norm;
    float inv_norm = 1.0f / norm;

    float rot_normalized[4];
    rot_normalized[0] = (v.rot0 * inv_norm) * 128.0f + 128.0f;
    rot_normalized[1] = (v.rot1 * inv_norm) * 128.0f + 128.0f;
    rot_normalized[2] = (v.rot2 * inv_norm) * 128.0f + 128.0f;
    rot_normalized[3] = (v.rot3 * inv_norm) * 128.0f + 128.0f;

    // Clamp and assign to output
    for (int i = 0; i < 4; ++i)
    {
        rot_normalized[i] = fminf(fmaxf(rot_normalized[i], 0.0f), 255.0f);
        d_output[idx].rot[i] = static_cast<uint8_t>(rot_normalized[i]);
    }

    // Compute sort key
    float sum_scale = v.scale0 + v.scale1 + v.scale2;
    float numerator = expf(sum_scale);
    float denom = 1.0f + expf(-v.opacity);
    d_sort_keys[idx] = -numerator / denom; // Negative for descending order
}

// Function to process a single PLY file and convert it to SPLAT format using CUDA
void process_ply_to_splat_cuda(const std::string &ply_file_path, const std::string &output_file)
{
    std::cout << "Reading PLY file: " << ply_file_path << std::endl;

    // Read the PLY file
    std::ifstream ifs(ply_file_path, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "Error: Cannot open input file " << ply_file_path << std::endl;
        return;
    }

    tinyply::PlyFile file;
    file.parse_header(ifs);

    // Extract vertex properties
    std::shared_ptr<tinyply::PlyData> vert_x = file.request_properties_from_element("vertex", { "x" });
    std::shared_ptr<tinyply::PlyData> vert_y = file.request_properties_from_element("vertex", { "y" });
    std::shared_ptr<tinyply::PlyData> vert_z = file.request_properties_from_element("vertex", { "z" });

    std::shared_ptr<tinyply::PlyData> vert_scale0 = file.request_properties_from_element("vertex", { "scale_0" });
    std::shared_ptr<tinyply::PlyData> vert_scale1 = file.request_properties_from_element("vertex", { "scale_1" });
    std::shared_ptr<tinyply::PlyData> vert_scale2 = file.request_properties_from_element("vertex", { "scale_2" });

    std::shared_ptr<tinyply::PlyData> vert_rot0 = file.request_properties_from_element("vertex", { "rot_0" });
    std::shared_ptr<tinyply::PlyData> vert_rot1 = file.request_properties_from_element("vertex", { "rot_1" });
    std::shared_ptr<tinyply::PlyData> vert_rot2 = file.request_properties_from_element("vertex", { "rot_2" });
    std::shared_ptr<tinyply::PlyData> vert_rot3 = file.request_properties_from_element("vertex", { "rot_3" });

    std::shared_ptr<tinyply::PlyData> vert_fdc0 = file.request_properties_from_element("vertex", { "f_dc_0" });
    std::shared_ptr<tinyply::PlyData> vert_fdc1 = file.request_properties_from_element("vertex", { "f_dc_1" });
    std::shared_ptr<tinyply::PlyData> vert_fdc2 = file.request_properties_from_element("vertex", { "f_dc_2" });

    std::shared_ptr<tinyply::PlyData> vert_opacity = file.request_properties_from_element("vertex", { "opacity" });

    file.read(ifs);

    if (!vert_x || !vert_y || !vert_z ||
        !vert_scale0 || !vert_scale1 || !vert_scale2 ||
        !vert_rot0 || !vert_rot1 || !vert_rot2 || !vert_rot3 ||
        !vert_fdc0 || !vert_fdc1 || !vert_fdc2 ||
        !vert_opacity)
    {
        std::cerr << "Error: Missing required vertex properties in " << ply_file_path << std::endl;
        return;
    }

    const size_t num_vertices = vert_x->count;
    std::vector<Vertex> vertices(num_vertices);

    // Helper lambda to convert a specific property
    auto read_floats = [](std::shared_ptr<tinyply::PlyData> d) -> const float* {
        return reinterpret_cast<const float*>(d->buffer.get());
    };

    const float *x_arr = read_floats(vert_x);
    const float *y_arr = read_floats(vert_y);
    const float *z_arr = read_floats(vert_z);

    const float *s0_arr = read_floats(vert_scale0);
    const float *s1_arr = read_floats(vert_scale1);
    const float *s2_arr = read_floats(vert_scale2);

    const float *r0_arr = read_floats(vert_rot0);
    const float *r1_arr = read_floats(vert_rot1);
    const float *r2_arr = read_floats(vert_rot2);
    const float *r3_arr = read_floats(vert_rot3);

    const float *fdc0_arr = read_floats(vert_fdc0);
    const float *fdc1_arr = read_floats(vert_fdc1);
    const float *fdc2_arr = read_floats(vert_fdc2);

    const float *op_arr = read_floats(vert_opacity);

    std::cout << "Populating vertex data..." << std::endl;

    // Populate the vertices vector
    #pragma omp parallel for
    for (size_t i = 0; i < num_vertices; i++)
    {
        vertices[i].x = x_arr[i];
        vertices[i].y = y_arr[i];
        vertices[i].z = z_arr[i];
        vertices[i].scale0 = s0_arr[i];
        vertices[i].scale1 = s1_arr[i];
        vertices[i].scale2 = s2_arr[i];
        vertices[i].rot0 = r0_arr[i];
        vertices[i].rot1 = r1_arr[i];
        vertices[i].rot2 = r2_arr[i];
        vertices[i].rot3 = r3_arr[i];
        vertices[i].f_dc_0 = fdc0_arr[i];
        vertices[i].f_dc_1 = fdc1_arr[i];
        vertices[i].f_dc_2 = fdc2_arr[i];
        vertices[i].opacity = op_arr[i];
    }

    std::cout << "Allocating GPU memory..." << std::endl;

    // Allocate device memory
    Vertex* d_vertices;
    OutputVertex* d_output;
    float* d_sort_keys;
    cudaCheckError(cudaMalloc(&d_vertices, num_vertices * sizeof(Vertex)));
    cudaCheckError(cudaMalloc(&d_output, num_vertices * sizeof(OutputVertex)));
    cudaCheckError(cudaMalloc(&d_sort_keys, num_vertices * sizeof(float)));

    std::cout << "Copying vertex data to GPU..." << std::endl;

    // Copy data to GPU
    cudaCheckError(cudaMemcpy(d_vertices, vertices.data(), num_vertices * sizeof(Vertex), cudaMemcpyHostToDevice));

    // Define CUDA kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_vertices + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching CUDA kernel..." << std::endl;

    // Launch CUDA kernel
    compute_vertices_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_vertices, d_output, d_sort_keys, num_vertices, 0.28209479177387814);
    cudaCheckError(cudaDeviceSynchronize());

    std::cout << "CUDA kernel execution completed." << std::endl;

    std::cout << "Performing GPU-based sorting with Thrust..." << std::endl;

    // Use Thrust to sort by sort_keys
    thrust::device_ptr<float> thrust_sort_keys(d_sort_keys);
    thrust::device_ptr<OutputVertex> thrust_output(d_output);

    // Perform sort_by_key: sort_keys ascending, which corresponds to descending in the original
    thrust::sort_by_key(thrust_sort_keys, thrust_sort_keys + num_vertices, thrust_output);

    std::cout << "Sorting completed." << std::endl;

    // Allocate host memory for sorted output
    std::vector<OutputVertex> sorted_output(num_vertices);

    std::cout << "Copying sorted data back to host..." << std::endl;

    // Copy sorted data back to host
    cudaCheckError(cudaMemcpy(sorted_output.data(), d_output, num_vertices * sizeof(OutputVertex), cudaMemcpyDeviceToHost));

    std::cout << "Data transfer completed." << std::endl;

    // Free device memory
    cudaCheckError(cudaFree(d_vertices));
    cudaCheckError(cudaFree(d_output));
    cudaCheckError(cudaFree(d_sort_keys));

    std::cout << "Writing sorted data to SPLAT file: " << output_file << std::endl;

    // Open output file for writing
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "Error: Cannot open output file " << output_file << std::endl;
        return;
    }

    // Write sorted data to the output file
    for (size_t i = 0; i < num_vertices; i++)
    {
        ofs.write(reinterpret_cast<const char*>(&sorted_output[i].position), sizeof(sorted_output[i].position));
        ofs.write(reinterpret_cast<const char*>(&sorted_output[i].scales), sizeof(sorted_output[i].scales));
        ofs.write(reinterpret_cast<const char*>(&sorted_output[i].color), sizeof(sorted_output[i].color));
        ofs.write(reinterpret_cast<const char*>(&sorted_output[i].rot), sizeof(sorted_output[i].rot));
    }

    ofs.close();
    std::cout << "Saved " << output_file << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " input_files... [--output out.splat]" << std::endl;
        return 1;
    }

    std::string output = "output.splat";
    std::vector<std::string> input_files;

    // Simple argument parsing
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if ((arg == "--output" || arg == "-o") && i + 1 < argc)
        {
            output = argv[++i];
        } 
        else
        {
            input_files.push_back(arg);
        }
    }

    if (input_files.empty())
    {
        std::cerr << "No input files provided." << std::endl;
        return 1;
    }

    if (input_files.size() == 1)
    {
        std::cout << "Processing " << input_files[0] << "..." << std::endl;
        process_ply_to_splat_cuda(input_files[0], output);
    } 
    else
    {
        for (auto &input_file : input_files)
        {
            std::cout << "Processing " << input_file << "..." << std::endl;
            std::string out_name = input_file + ".splat";
            process_ply_to_splat_cuda(input_file, out_name);
        }
    }

    return 0;
}
