#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "tinyply.h"
#include <iomanip>
#include <filesystem>
#include <pybind11/pybind11.h>


namespace fs = std::filesystem;

struct Vertex
{
    float x, y, z;
    float scale0, scale1, scale2;
    float rot0, rot1, rot2, rot3;
    float f_dc_0, f_dc_1, f_dc_2;
    float opacity;

    // -exp(scale_0 + scale_1 + scale_2) / (1 + exp(-opacity))
    // we can store that once per vertex to avoid recomputing during sort.
    float sort_key() const
    {
        float sum_scale = scale0 + scale1 + scale2;
        // negative for descending order
        return -std::exp(sum_scale) / (1.0f + std::exp(-opacity)); 
    }
};

static const double SH_C0 = 0.28209479177387814;

// normalizing a quaternion (rot0, rot1, rot2, rot3) and transform it
// into [0,255]-range bytes
void normalize_and_transform_rot(const float *rot, uint8_t *out)
{
    float norm = std::sqrt(rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2] + rot[3]*rot[3]);
    // avoid division by zero
    if (norm < 1e-12f) norm = 1e-12f; 
    float inv_norm = 1.0f / norm;

    for (int i = 0; i < 4; i++)
    {
        float val = (rot[i] * inv_norm) * 128.0f + 128.0f;
        if (val < 0.0f) val = 0.0f; 
        else if (val > 255.0f) val = 255.0f;
        out[i] = static_cast<uint8_t>(val);
    }
}

void float_to_bytes(const float *val, size_t count, std::ofstream &out)
{
    out.write(reinterpret_cast<const char*>(val), count * sizeof(float));
}

void process_ply_to_splat(const std::string &ply_file_path, const std::string &output_file)
{
    // reading the PLY file
    std::ifstream ifs(ply_file_path, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "Error: Cannot open input file " << ply_file_path << std::endl;
        return;
    }

    tinyply::PlyFile file;
    file.parse_header(ifs);

    // extracting vertex properties.
    std::shared_ptr<tinyply::PlyData> vert_x = file.request_properties_from_element("vertex", {"x"});
    std::shared_ptr<tinyply::PlyData> vert_y = file.request_properties_from_element("vertex", {"y"});
    std::shared_ptr<tinyply::PlyData> vert_z = file.request_properties_from_element("vertex", {"z"});

    std::shared_ptr<tinyply::PlyData> vert_scale0 = file.request_properties_from_element("vertex", {"scale_0"});
    std::shared_ptr<tinyply::PlyData> vert_scale1 = file.request_properties_from_element("vertex", {"scale_1"});
    std::shared_ptr<tinyply::PlyData> vert_scale2 = file.request_properties_from_element("vertex", {"scale_2"});

    std::shared_ptr<tinyply::PlyData> vert_rot0 = file.request_properties_from_element("vertex", {"rot_0"});
    std::shared_ptr<tinyply::PlyData> vert_rot1 = file.request_properties_from_element("vertex", {"rot_1"});
    std::shared_ptr<tinyply::PlyData> vert_rot2 = file.request_properties_from_element("vertex", {"rot_2"});
    std::shared_ptr<tinyply::PlyData> vert_rot3 = file.request_properties_from_element("vertex", {"rot_3"});

    std::shared_ptr<tinyply::PlyData> vert_fdc0 = file.request_properties_from_element("vertex", {"f_dc_0"});
    std::shared_ptr<tinyply::PlyData> vert_fdc1 = file.request_properties_from_element("vertex", {"f_dc_1"});
    std::shared_ptr<tinyply::PlyData> vert_fdc2 = file.request_properties_from_element("vertex", {"f_dc_2"});

    std::shared_ptr<tinyply::PlyData> vert_opacity = file.request_properties_from_element("vertex", {"opacity"});

    file.read(ifs);

    if (!vert_x       || 
        !vert_y       || 
        !vert_z       ||
        !vert_scale0  || 
        !vert_scale1  || 
        !vert_scale2  ||
        !vert_rot0    || 
        !vert_rot1    || 
        !vert_rot2    || 
        !vert_rot3    ||
        !vert_fdc0    || 
        !vert_fdc1    || 
        !vert_fdc2    ||
        !vert_opacity)
    {
        std::cerr << "Error: Missing required vertex properties in " << ply_file_path << std::endl;
        return;
    }

    const size_t num_vertices = vert_x->count;
    std::vector<Vertex> vertices;
    vertices.reserve(num_vertices);

    // helper lambda to convert a specific property
    auto read_floats = [](std::shared_ptr<tinyply::PlyData> d)
    {
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

    for (size_t i = 0; i < num_vertices; i++)
    {
        Vertex v;
        v.x = x_arr[i]; 
        v.y = y_arr[i]; 
        v.z = z_arr[i];
        v.scale0 = s0_arr[i]; 
        v.scale1 = s1_arr[i]; 
        v.scale2 = s2_arr[i];
        v.rot0 = r0_arr[i]; 
        v.rot1 = r1_arr[i]; 
        v.rot2 = r2_arr[i]; 
        v.rot3 = r3_arr[i];
        v.f_dc_0 = fdc0_arr[i]; 
        v.f_dc_1 = fdc1_arr[i]; 
        v.f_dc_2 = fdc2_arr[i];
        v.opacity = op_arr[i];
        vertices.push_back(v);
    }

    // sort vertices by the computed key
    std::sort(vertices.begin(), vertices.end(), [](const Vertex &a, const Vertex &b)
    {
        return a.sort_key() < b.sort_key(); 
    });

    // open output file for writing
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "Error: Cannot open output file " << output_file << std::endl;
        return;
    }

    // writing data in required format
    // for each vertex:
    //  position: 3 floats
    //  scales: exp of each scale_*
    //  color: [ (0.5 + SH_C0*f_dc_*)*255, last is 1/(1+exp(-opacity))*255 ]
    //  rot: normalized quaternion mapped to [0,255]
    for (const auto &v : vertices)
    {
        float position[3] = { v.x, v.y, v.z };
        float scales[3] = { std::exp(v.scale0), std::exp(v.scale1), std::exp(v.scale2) };

        float denominator = 1.0f + std::exp(-v.opacity);
        float alpha = 1.0f / denominator;

        float color_f[4] =
        {
            0.5f + static_cast<float>(SH_C0) * v.f_dc_0,
            0.5f + static_cast<float>(SH_C0) * v.f_dc_1,
            0.5f + static_cast<float>(SH_C0) * v.f_dc_2,
            alpha
        };

        // converting color to [0,255]
        uint8_t color_u8[4];
        for (int i = 0; i < 4; i++)
        {
            float c = color_f[i] * 255.0f;
            if (c < 0.0f) c = 0.0f;
            else if (c > 255.0f) c = 255.0f;
            color_u8[i] = static_cast<uint8_t>(c);
        }

        float rot[4] = { v.rot0, v.rot1, v.rot2, v.rot3 };
        uint8_t rot_u8[4];
        normalize_and_transform_rot(rot, rot_u8);

        // writing binary data
        ofs.write(reinterpret_cast<const char*>(position), sizeof(position));
        ofs.write(reinterpret_cast<const char*>(scales),   sizeof(scales));
        ofs.write(reinterpret_cast<const char*>(color_u8), sizeof(color_u8));
        ofs.write(reinterpret_cast<const char*>(rot_u8),   sizeof(rot_u8));
    }

    ofs.close();
}


PYBIND11_MODULE(ply2splat, m)
{
    m.doc() = "PLY to SPLAT converter module";
    m.def("process_ply_to_splat", &process_ply_to_splat, "Convert a PLY file to a SPLAT file",
          pybind11::arg("ply_file_path"),
          pybind11::arg("output_file"));
}


int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " input_files..." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::string> input_files;

    // simple arg parsing to collect all input files
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        input_files.push_back(arg);
    }

    if (input_files.empty())
    {
        std::cerr << "No input files provided." << std::endl;
        return EXIT_FAILURE;
    }

    // iterate over each input file
    for (const auto &input_file : input_files)
    {
        std::cout << "Processing Output Data Encoding for " << input_file << "..." << std::endl;

        // generating output file name by replacing the last extension with ".splat"
        size_t last_dot = input_file.find_last_of('.');
        std::string output_file;
        if (last_dot != std::string::npos)
        {
            output_file = input_file.substr(0, last_dot) + ".splat";
        }
        else
        {
            // if no extension is found, append ".splat"
            output_file = input_file + ".splat";
        }

        process_ply_to_splat(input_file, output_file);

        try
        {
            // getting input and output files sizes
            uintmax_t input_size_bytes = fs::file_size(input_file);
            uintmax_t output_size_bytes = fs::file_size(output_file);

            // convert bytes to MB ( 1024^2 )
            const double bytes_to_mb = 1024.0 * 1024.0;
            double input_size_mb = static_cast<double>(input_size_bytes) / bytes_to_mb;
            double output_size_mb = static_cast<double>(output_size_bytes) / bytes_to_mb;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Input file size: " << input_size_mb << " MB" << std::endl;
            std::cout << "Output file size: " << output_size_mb << " MB" << std::endl;
        }
        catch (const fs::filesystem_error &e)
        {
            std::cerr << "Error accessing file sizes: " << e.what() << std::endl;
        }

        std::cout << "----------------------------------------" << std::endl;
    }

    return EXIT_SUCCESS;
}
