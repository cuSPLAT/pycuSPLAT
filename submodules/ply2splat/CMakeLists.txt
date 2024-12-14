cmake_minimum_required(VERSION 3.12)

project(ply2splat)

# Enable Pybind11's modern Python support
set(PYBIND11_FINDPYTHON ON)

# Find the Pybind11 package
find_package(pybind11 REQUIRED)

# Define the target for the Python module
pybind11_add_module(
    ply2splat
    ply2splat_bindings.cpp
    tinyply.cpp  # Ensure tinyply.cpp is included directly here
)

# Add include directories for tinyply if it requires specific headers
target_include_directories(ply2splat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set the C++ standard to C++17
set_target_properties(ply2splat PROPERTIES CXX_STANDARD 17)

# Add additional compile options for optimization
target_compile_options(ply2splat PRIVATE -O3 -Wall -Wextra)