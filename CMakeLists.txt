cmake_minimum_required(VERSION 3.15)

project(kllm VERSION 0.1.0 LANGUAGES CXX)

# Header-only interface target
add_library(kllm INTERFACE)

target_include_directories(kllm
	INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(kllm INTERFACE cxx_std_17)

# Be moderately strict about warnings for our targets
if(NOT MSVC)
	target_compile_options(kllm INTERFACE -Wall -Wextra -Wpedantic -O3)
else()
	target_compile_options(kllm INTERFACE /W4)
endif()

# Demo executable
add_executable(kllm_demo examples/main.cpp)

target_link_libraries(kllm_demo PRIVATE kllm)